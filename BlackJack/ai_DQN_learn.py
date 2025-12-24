import torch
import torch.nn as nn
import copy
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from NN_structure import BJNet

# === カウンティング用グローバル変数 ===
N_TOTAL_CARDS_INIT = N_DECKS * 52
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10

### グローバル変数 ###
g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None

# Q学習の設定値
EPS_START = 1.0  # 学習初期はランダム探索を多くする
EPS_END = 0.1    # 最終的なランダム探索率
EPS_DECAY = 20000  # ε減衰のスピード
EPS = EPS_START

LEARNING_RATE = 0.0001 # 学習率
DISCOUNT_FACTOR = 0.95 # 割引率
BATCH_SIZE = 128
BUFFER_SIZE = 100000

# モデルとオプティマイザの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    q_net = BJNet().to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
except Exception as e:
    print(f"Model Init Error: {e}")
    q_net = None

# 行動の定義（ニューラルネットワークの出力順序に対応）
ACTION_LIST = [
    Action.HIT,
    Action.STAND,
    Action.DOUBLE_DOWN,
    Action.SURRENDER,
    Action.RETRY
]

def action_to_idx(action):
    try:
        return ACTION_LIST.index(action)
    except ValueError:
        return 0 # UNDEFINED等の場合は一旦0扱い

def idx_to_action(idx):
    return ACTION_LIST[idx]

# === Replay Buffer (ai_yosssi.pyより移植) ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state等は既にnumpy配列等を想定
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# グローバルバッファ
replay_buffer = ReplayBuffer(BUFFER_SIZE)


### カードカウンティング関数 ###
def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_info):
    global g_card_counter, g_total_cards_seen
    
    # カード情報が整数の場合（初期配布時など）
    if isinstance(card_info, int):
        rank_idx = (card_info % 13)
    # 文字列の場合（'Heart-Q'など）
    elif isinstance(card_info, str):
        if card_info == 'X' or card_info is None: return
        try:
            parts = card_info.split('-')
            if len(parts) < 2: return
            card_rank_str = parts[1] 
            
            if card_rank_str == 'A': rank_idx = 0
            elif card_rank_str == 'J': rank_idx = 10
            elif card_rank_str == 'Q': rank_idx = 11
            elif card_rank_str == 'K': rank_idx = 12
            else: rank_idx = int(card_rank_str) - 1 
        except:
            return
    else:
        return

    # カウント更新
    if 0 <= rank_idx < 13 and g_card_counter[rank_idx] > 0:
        g_card_counter[rank_idx] -= 1
        g_total_cards_seen += 1

### ゲーム進行関数 ###

def game_start(game_ID=0):
    global g_retry_counter, player, soc

    print('Game {0} start.'.format(game_ID))
    g_retry_counter = 0

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bet, money = player.set_bet()
    print(f'Action: BET (money: {money}, bet: {bet})')

    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        print('Dealer said: Card set has been shuffled.')
        initialize_card_counter()

    # 初期カード受信 & カウント
    dc, pc1, pc2 = player.receive_init_cards(soc)
    
    # ★修正点1: 初期手札をカウントに反映
    update_card_counter(dc)
    update_card_counter(pc1)
    update_card_counter(pc2)

    print('Delaer gave cards.')
    print('  dealer-card: ', get_card_info(dc))
    print('  player-card 1: ', get_card_info(pc1))
    print('  player-card 2: ', get_card_info(pc2))

def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# 各アクション関数（カウンティング更新付き）
def hit():
    print('Action: HIT')
    player.send_message(soc, 'hit')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc))
    print('  player-card: ', get_card_info(pc))
    
    if status == 'bust':
        for c in dc: update_card_counter(get_card_info(c)) # ディーラーカードもカウント
        soc.close()
        reward = player.update_money(rate=rate)
        return reward, True, status
    return 0, False, status

def stand():
    print('Action: STAND')
    player.send_message(soc, 'stand')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for c in dc: update_card_counter(get_card_info(c))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def double_down():
    print('Action: DOUBLE DOWN')
    player.double_bet()
    player.send_message(soc, 'double_down')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc))
    for c in dc: update_card_counter(get_card_info(c))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def surrender():
    print('Action: SURRENDER')
    player.send_message(soc, 'surrender')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for c in dc: update_card_counter(get_card_info(c))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def retry():
    print('Action: RETRY')
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    update_card_counter(get_card_info(pc)) # 新しいカードをカウント
    
    if status == 'bust':
        for c in dc: update_card_counter(get_card_info(c))
        soc.close()
        reward = player.update_money(rate=rate)
        return reward - penalty, True, status
    return -penalty, False, status

def act(action: Action):
    if action == Action.HIT: return hit()
    elif action == Action.STAND: return stand()
    elif action == Action.DOUBLE_DOWN: return double_down()
    elif action == Action.SURRENDER: return surrender()
    elif action == Action.RETRY: return retry()
    else: exit()

# === 状態取得（正規化対応版） ===
def get_state():
    p_hand, d_hand = get_current_hands()
    
    score = p_hand.get_score() / 30.0 # 正規化: スコアは大体30以下
    length = p_hand.length() / 10.0   # 正規化: 枚数は10以下
    d_score = d_hand.get_score() / 30.0

    # ソフトハンド（エースを11として使えるか）
    has_ace = False
    raw_score = 0
    for card_id in p_hand.cards:
        rank = (card_id % 13) + 1
        if rank == 1: has_ace = True
        raw_score += min(10, rank)
    soft_hand_val = 1.0 if (has_ace and raw_score + 10 <= 21) else 0.0

    # ★修正点2: カウンティングの正規化
    # 残りカード枚数
    remaining_cards = max(1, N_TOTAL_CARDS_INIT - g_total_cards_seen)
    # 確率に変換 (count / remaining)
    normalized_counter = g_card_counter.astype(np.float32) / remaining_cards

    # 結合 (4要素 + 13要素 = 17要素)
    state_vector = np.concatenate([
        np.array([score, length, soft_hand_val, d_score]), 
        normalized_counter
    ]).astype(np.float32)
    
    return state_vector # numpy配列として返す（バッファに入れるため）

# === 行動選択 ===
def select_action(state, testmode=False):
    global q_net, EPS
    
    # 探索 (Epsilon-Greedy)
    if not testmode and random.random() < EPS:
        return np.random.choice(ACTION_LIST)
    
    q_net.eval() # 評価モードに切り替え

    # 活用 (Argmax Q)
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = q_net(state_tensor)
        action_idx = torch.argmax(q_values).item()
    
    if not testmode:
        q_net.train() # 学習モードに戻す
    
    return idx_to_action(action_idx)

# === 学習ステップ (ai_yosssi.pyベース) ===
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # 現在のQ値 Q(s, a)
    q_values = q_net(states)
    # actionsはインデックス(0-4)のtensorである必要あり
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 次のQ値 max Q(s', a')
    with torch.no_grad():
        next_q_values = q_net(next_states).max(1)[0]
        target_q = rewards + DISCOUNT_FACTOR * next_q_values * (1 - dones)

    # ロス計算と更新
    loss = nn.MSELoss()(current_q, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

### メイン処理 ###
def main():
    global g_retry_counter, EPS

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--history', type=str, default='play_log.csv')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--testmode', action='store_true')
    args = parser.parse_args()

    n_games = args.games + 1
    
    if args.load != '':
        try:
            q_net.load_state_dict(torch.load(args.load))
            print("Model loaded.")
        except:
            print("Failed to load model.")

    if args.testmode:
        q_net.eval()
        EPS = 0.0
    
    money_history = []
    loss_history = []
    win_count = 0
    
    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile)

    initialize_card_counter() # 最初の初期化

    for n in range(1, n_games):
        game_start(n)
        state = get_state()
        
        # 線形減衰のコード
        # n が EPS_DECAY に達したら、それ以降はずっと EPS_END にする
        if n < EPS_DECAY:
            # 割合を計算 (0.0 -> 1.0)
            ratio = n / EPS_DECAY
            # スタートから徐々に減らす
            EPS = EPS_START - (EPS_START - EPS_END) * ratio
        else:
            # 20000回以降は最小値で固定
            EPS = EPS_END

        while True:
            action = select_action(state, args.testmode)
            
            # RETRY制限時の強制変更
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                action = np.random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])
            
            # 行動実行
            reward, done, status = act(action)
            if action == Action.RETRY: g_retry_counter += 1

            next_state = get_state()

            # バッファに追加 & 学習
            if not args.testmode:
                action_idx = action_to_idx(action)

                # === 【修正】報酬の正規化 ===
                # 金額をベット額で割って、およそ -1.0 〜 +1.5 の範囲に収める
                # (player.basic_bet は config.py の BET と同じ値です)
                normalized_reward = reward / player.basic_bet
                
                # bufferには reward ではなく normalized_reward を入れる
                replay_buffer.push(state, action_idx, normalized_reward, next_state, done)
                
                loss = train_step()
                if loss is not None: loss_history.append(loss)

            print(f'{state[0]*30:.0f},{state[1]*10:.0f},{get_action_name(action)},{status},{reward}', file=logfile)
            
            state = next_state

            if done:
                money_history.append(player.get_money())
                if status == 'win' or status == 'dealer_bust':
                    win_count += 1
                break
        print('')

    logfile.close()
    
    # 結果表示
    print(f"Total Games: {args.games}, Wins: {win_count}, Win Rate: {win_count/args.games:.2%}")
    
    if args.save != '':
        torch.save(q_net.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    # グラフ描画
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(money_history)
    plt.title("Money History")
    plt.grid()
    
    if not args.testmode and loss_history:
        plt.subplot(1, 2, 2)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.grid()
    
    plt.show()

if __name__ == '__main__':
    main()