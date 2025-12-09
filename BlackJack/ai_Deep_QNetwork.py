# --- 必要なライブラリ ---
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import socket
import argparse

# パス解決のための「おまじない」
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque 

# グラフ描画用
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- プロジェクト共通のコンポーネント ---
from .classes import Action, Player, get_card_info, get_action_name
from .config import PORT, BET, INITIAL_MONEY, N_DECKS, SHUFFLE_INTERVAL, SHUFFLE_THRESHOLD
from mylib.utility import print_args
from .NN_structure import BJNet

# --- DQNのハイパーパラメータ（長期学習・高精度用） ---
REPLAY_BUFFER_SIZE = 100000 # 記憶できる経験を増やす
BATCH_SIZE = 128            # 一度に学習するデータを増やす
DISCOUNT_FACTOR = 0.99      # 将来の報酬もしっかり考慮
LEARNING_RATE = 0.00005      # ゆっくり、確実に学習する
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_GAMES = 40000     # 2万回くらいかけてじっくり探検させる
TARGET_UPDATE_FREQ = 100    # ターゲット更新は少しゆっくりに  

# --- グローバル変数 ---
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None
nn_model = None         
target_model = None     
optimizer = None
loss_func = None
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
g_device = ''
g_epsilon = EPS_START
action_set = [Action.DOUBLE_DOWN, Action.HIT, Action.RETRY, Action.STAND, Action.SURRENDER]
RETRY_MAX = 10 
g_retry_counter = 0
g_prev_player_cards = set() 
g_prev_dealer_card = 'X'    

MODEL_DIR = './BJNet_models_DQN' 

# === カードカウンティング機能 ===
N_TOTAL_CARDS = N_DECKS * 52
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_id_str):
    global g_card_counter, g_total_cards_seen
    if card_id_str == 'X' or card_id_str is None: 
        return
    
    try:
        if isinstance(card_id_str, int):
             rank_idx = (card_id_str % 13)
        elif card_id_str.isdigit():
             cid = int(card_id_str)
             rank_idx = (cid % 13)
        else:
            parts = card_id_str.split('-')
            if len(parts) < 2: return
            card_rank_str = parts[1] 
            
            if card_rank_str == 'A': rank_idx = 0
            elif card_rank_str == 'J': rank_idx = 10
            elif card_rank_str == 'Q': rank_idx = 11
            elif card_rank_str == 'K': rank_idx = 12
            else: rank_idx = int(card_rank_str) - 1 

        if g_card_counter[rank_idx] > 0: 
            g_card_counter[rank_idx] -= 1
            g_total_cards_seen += 1
            
    except Exception as e:
        pass

# === 状態ベクトルの定義 ===
def get_state(done=False):
    global player, g_card_counter, g_total_cards_seen

    score = player.player_hand.get_score()
    n_cards = player.player_hand.length()
    
    has_ace = False
    raw_score_assuming_ace_is_1 = 0
    for card_id in player.player_hand.cards:
        rank = (card_id % 13) + 1 
        if rank == 1:
            has_ace = True
            raw_score_assuming_ace_is_1 += 1
        else:
            raw_score_assuming_ace_is_1 += min(10, rank)
    
    if has_ace and (raw_score_assuming_ace_is_1 + 10 <= 21):
        soft_hand_val = 1.0
    else:
        soft_hand_val = 0.0

    if len(player.dealer_hand.cards) > 0:
        dealer_card_id = player.dealer_hand.cards[0]
        dealer_open_card_score = min(10, (dealer_card_id % 13) + 1)
    else:
        dealer_open_card_score = 0
    
    state_vector = np.concatenate([
        np.array([score, n_cards, soft_hand_val, dealer_open_card_score]), 
        g_card_counter.astype(np.float32)
    ]).astype(np.float32)
    
    return torch.from_numpy(state_vector).unsqueeze(0)


# === DQNエージェントの行動選択 ===
def select_action(state_tensor):
    global g_device, nn_model, g_epsilon, args

    if not args.testmode and np.random.rand() < g_epsilon:
        action = np.random.choice([
            Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
        ])
    else:
        nn_model.eval() 
        with torch.inference_mode():
            q_values = nn_model(state_tensor.to(g_device))
            action_index = torch.argmax(q_values).item()
            action = action_set[action_index]
    
    if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
        action = np.random.choice([Action.HIT, Action.STAND]) 

    return action


# === DQNの学習ロジック ===
def train_nn():
    global nn_model, target_model, optimizer, loss_func, replay_buffer, g_device

    if len(replay_buffer) < BATCH_SIZE:
        return None # 学習しなかった場合はNoneを返す

    nn_model.train() 
    
    batch = random.sample(replay_buffer, BATCH_SIZE)

    state_batch = torch.cat([s for (s, a, r, n_s, d) in batch]).to(g_device)
    action_batch = torch.tensor([a for (s, a, r, n_s, d) in batch], dtype=torch.int64).unsqueeze(1).to(g_device)
    reward_batch = torch.tensor([r for (s, a, r, n_s, d) in batch], dtype=torch.float32).unsqueeze(1).to(g_device)
    next_state_batch = torch.cat([n_s for (s, a, r, n_s, d) in batch]).to(g_device)
    done_batch = torch.tensor([d for (s, a, r, n_s, d) in batch], dtype=torch.float32).unsqueeze(1).to(g_device)

    q_values = nn_model(state_batch)
    q_s_a = q_values.gather(1, action_batch) 

    with torch.no_grad():
        next_q_values = target_model(next_state_batch)
        max_next_q = next_q_values.max(1, keepdim=True)[0]
        max_next_q[done_batch == 1.0] = 0.0 
        target_q_s_a = reward_batch + (DISCOUNT_FACTOR * max_next_q)

    loss = loss_func(q_s_a, target_q_s_a)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() # lossの値を返す


# === 通信関連の関数 ===
def connect_sv(port):
    global soc
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        soc.connect((socket.gethostname(), port))
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # print('Connected to the dealer.')
    except Exception as e:
        print(f"Failed to connect to dealer: {e}")
        sys.exit(1)

def game_start(game_ID=0):
    global g_retry_counter, player, soc
    g_retry_counter = 0 
    bet, money = player.set_bet()

def game_end():
    global soc
    soc.close()

def act(action):
    global player, soc, g_retry_counter

    if action == Action.HIT: cmd = 'hit'
    elif action == Action.STAND: cmd = 'stand'
    elif action == Action.DOUBLE_DOWN: cmd = 'double_down'
    elif action == Action.SURRENDER: cmd = 'surrender'
    elif action == Action.RETRY: cmd = 'retry'
    else: cmd = 'stand'

    player.send_message(soc, cmd)

    try:
        if action == Action.HIT:
            pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
            update_card_counter(get_card_info(pc))
            
        elif action == Action.DOUBLE_DOWN:
            pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
            update_card_counter(get_card_info(pc))
            
        elif action == Action.RETRY:
            pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
            update_card_counter(get_card_info(pc))
            g_retry_counter += 1
            
        elif action == Action.STAND or action == Action.SURRENDER:
            score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
            
        else:
            return 0.0, True, 'error', 0.0

    except Exception as e:
        return 0.0, True, 'error', 0.0

    if 'dc' in locals():
        if len(dc) > 1:
            for i in range(1, len(dc)):
                update_card_counter(get_card_info(dc[i]))

    learning_reward = 0.0
    if status == 'win' or status == 'dealer_bust':
        learning_reward = 1.0
    elif status == 'lose':
        learning_reward = -1.0
    elif status == 'bust':
        learning_reward = -5.0
    elif status == 'push' or status == 'draw':
        learning_reward = 0.0
    elif status == 'surrendered':
        learning_reward = -0.5
    
    if action == Action.RETRY:
        learning_reward -= 0.1

    final_reward = 0.0
    done = (status != 'unsettled')
    if done:
        final_reward = player.update_money(rate=rate)

    return final_reward, done, status, learning_reward


# === メインの実行ブロック ===
def main():
    global g_device, nn_model, target_model, optimizer, loss_func, g_epsilon, player, args

    parser = argparse.ArgumentParser(description='DQN AI Player for Blackjack')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID')
    parser.add_argument('--games', type=int, default=1000, help='num. of games to play')
    parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'dqn_model.pth'), type=str, help='file path of trained model')
    parser.add_argument('--testmode', help='run without learning', action='store_true')
    args_dict = print_args(parser.parse_args()) 
    args = argparse.Namespace(**args_dict) 

    if torch.cuda.is_available() and args.gpu >= 0:
        g_device = 'cuda:{0}'.format(args.gpu)
    else:
        g_device = 'cpu'
    print('Using device: {0}'.format(g_device))
    
    os.makedirs(MODEL_DIR, exist_ok=True) 

    try:
        nn_model = BJNet().to(g_device) 
        target_model = BJNet().to(g_device) 
    except Exception as e:
        print("Model Init Error")
        return

    if os.path.exists(args.model):
        try:
            nn_model.load_state_dict(torch.load(args.model, map_location=g_device))
            print('Loaded trained model.')
        except:
            print('Failed to load model, starting from scratch.')
    
    target_model.load_state_dict(nn_model.state_dict()) 
    
    optimizer = optim.Adam(nn_model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()
    total_games = args.games
    total_wins = 0
    initial_money = player.get_money()

    # 履歴
    history_games = []
    history_money = []
    history_win_rate = []
    history_loss = [] # ★追加: Lossの履歴

    print(f"Start playing {total_games} games...")

    for game_ID in range(1, total_games + 1):
        
        connect_sv(PORT)
        game_start(game_ID)
        
        # 1. シャッフル確認
        cardset_shuffled = player.receive_card_shuffle_status(soc)
        if cardset_shuffled:
            initialize_card_counter() 
            
        # 2. 初期カード受信
        dc, pc1, pc2 = player.receive_init_cards(soc)
        update_card_counter(get_card_info(dc))
        update_card_counter(get_card_info(pc1))
        update_card_counter(get_card_info(pc2))
             
        # 3. 状態取得
        state_tensor = get_state(done=False)
        
        # このゲーム内でのLossを記録するリスト
        game_losses = []

        while True: 
            action = select_action(state_tensor)
            final_reward, done, status, learning_reward = act(action) 
            next_state_tensor = get_state(done)
            
            if not args.testmode:
                action_index = action_set.index(action)
                replay_buffer.append((state_tensor, action_index, learning_reward, next_state_tensor, done))
                
                # ★修正: train_nnの戻り値(loss)を受け取る
                loss_val = train_nn()
                if loss_val is not None:
                    game_losses.append(loss_val)
                
            state_tensor = next_state_tensor 

            if done:
                if status == 'win':
                    total_wins += 1
                break
        
        game_end()
        
        g_epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (game_ID / EPS_DECAY_GAMES))

        if not args.testmode and game_ID % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(nn_model.state_dict())

        # ★追加: このゲームの平均Lossを記録
        if not args.testmode:
            if game_losses:
                history_loss.append(np.mean(game_losses))
            else:
                # 学習が行われなかった場合（バッファ不足など）は直前の値を入れるか0を入れる
                if history_loss:
                    history_loss.append(history_loss[-1])
                else:
                    history_loss.append(0)

        # 履歴記録
        if game_ID % 100 == 0:
            current_money = player.get_money()
            win_rate = (total_wins / game_ID) * 100
            history_games.append(game_ID)
            history_money.append(current_money)
            history_win_rate.append(win_rate)
            
            # Lossの表示用（直近100ゲームの平均）
            recent_loss = 0
            if not args.testmode and len(history_loss) > 0:
                recent_loss_avg = np.mean(history_loss[-100:])
            else:
                recent_loss_avg = 0

            print(f"Game {game_ID}: Money={current_money}, WinRate={win_rate:.2f}%, Loss={recent_loss_avg:.4f}, Epsilon={g_epsilon:.4f}")
            
            if not args.testmode and game_ID % 5000 == 0:
                torch.save(nn_model.state_dict(), os.path.join(MODEL_DIR, f'dqn_model_game{game_ID}.pth'))

    # 終了処理
    final_money = player.get_money()
    print("\n--- All games finished ---")
    print(f"Final Money: {final_money}$ (Total Profit: {final_money - initial_money}$)")
    print(f"Overall Win Rate: {(total_wins / total_games) * 100:.2f}%")
    
    if not args.testmode:
        torch.save(nn_model.state_dict(), args.model)

    # グラフ作成（★修正: Lossも含めて3つ表示）
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(18, 5)) # 横長にする
        
        # 1. Money
        plt.subplot(1, 3, 1)
        plt.plot(history_games, history_money)
        plt.axhline(y=initial_money, color='r', linestyle='--')
        plt.title('Money History')
        plt.xlabel('Games')
        plt.ylabel('Money')
        plt.grid(True)
        
        # 2. Win Rate
        plt.subplot(1, 3, 2)
        plt.plot(history_games, history_win_rate, color='orange')
        plt.title('Win Rate History')
        plt.xlabel('Games')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)

        # 3. Loss (★追加)
        plt.subplot(1, 3, 3)
        if not args.testmode:
            # ゲームごとのLossは変動が激しいので、history_gamesに対応するデータを作る
            # history_loss は全ゲーム分あるので、プロット用に間引くか、そのまま描画するか
            # ここでは見やすくするために移動平均をとって全ゲーム分描画
            
            # 移動平均の計算
            window = 100
            if len(history_loss) >= window:
                moving_avg = np.convolve(history_loss, np.ones(window)/window, mode='valid')
                plt.plot(range(window, len(history_loss)+1), moving_avg, color='green')
            else:
                plt.plot(range(1, len(history_loss)+1), history_loss, color='green')
                
            plt.title('Loss History (Moving Avg)')
            plt.xlabel('Games')
            plt.ylabel('Loss')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Test Mode (No Training)', ha='center')

        plt.tight_layout()
        
        # 保存
        graph_path = os.path.join(MODEL_DIR, 'history_graph_with_loss.png')
        plt.savefig(graph_path)
        print(f"Graph saved to {graph_path}")
        
        plt.show()

if __name__ == '__main__':
    main()