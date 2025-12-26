import copy
import socket
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# ai_player_Q.pyのインポートをそのまま使用
from classes import Action, Strategy, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS

# === スライドに基づくDQNモデル定義 (SimpleDQN) ===
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        # スライドの「ネットワーク構造」にあるFC層を再現（ただしDropout/BNはDQN用に除外）
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10

### グローバル変数 ###
g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None

# === カウンティング用変数 ===
# スライド「カウンティングの実装」に基づく
N_TOTAL_CARDS_INIT = N_DECKS * 52
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

# === DQNハイパーパラメータ (スライドの「学習パラメータの設定」を参考) ===
BATCH_SIZE = 128            # スライド: 128
GAMMA = 0.95                # スライド: 0.95 (バランス型)
EPS_START = 1.0
EPS_END = 0.1               # 最終的な探索率
EPS_DECAY = 20000           # スライド: 20000
TARGET_UPDATE = 500         # ターゲットネットワークの更新頻度
LEARNING_RATE = 0.0001      # スライド: 0.0001
MEMORY_CAPACITY = 100000    # スライド: 100000

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル構築 (入力17次元 -> 出力5次元)
policy_net = SimpleDQN(17, 5).to(device)
target_net = SimpleDQN(17, 5).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_CAPACITY)
steps_done = 0

# 行動リスト
ACTION_LIST = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]


### カウンティング関数 (新規追加) ###
def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_info):
    global g_card_counter, g_total_cards_seen
    rank_idx = -1
    
    # 整数の場合（Dealer.pyからの初期配布など）
    if isinstance(card_info, int):
        rank_idx = (card_info % 13)
    # 文字列の場合（get_card_infoの戻り値など）
    elif isinstance(card_info, str):
        if card_info == 'X' or card_info is None: return
        try:
            parts = card_info.split('-')
            if len(parts) < 2: return
            rank_str = parts[1] 
            if rank_str == 'A': rank_idx = 0
            elif rank_str == 'J': rank_idx = 10
            elif rank_str == 'Q': rank_idx = 11
            elif rank_str == 'K': rank_idx = 12
            else: rank_idx = int(rank_str) - 1 
        except: return
    else: return

    if 0 <= rank_idx < 13 and g_card_counter[rank_idx] > 0:
        g_card_counter[rank_idx] -= 1
        g_total_cards_seen += 1

### 関数 (ai_player_Q.pyベースでカウンティング更新を追加) ###

def game_start(game_ID=0):
    global g_retry_counter, player, soc
    print('Game {0} start.'.format(game_ID))
    g_retry_counter = 0
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bet, money = player.set_bet()
    print(f'Action: BET (money: {money}, bet: {bet})')

    # シャッフル時のカウンティングリセット
    if player.receive_card_shuffle_status(soc):
        print('Dealer said: Card set has been shuffled.')
        initialize_card_counter()

    dc, pc1, pc2 = player.receive_init_cards(soc)
    # ★追加: 初期カードをカウント
    update_card_counter(dc)
    update_card_counter(pc1)
    update_card_counter(pc2)

    print('Delaer gave cards.')
    print('  dealer-card: ', get_card_info(dc))
    print('  player-card 1: ', get_card_info(pc1))
    print('  player-card 2: ', get_card_info(pc2))

def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

def hit():
    print('Action: HIT')
    player.send_message(soc, 'hit')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc)) # ★追加
    print('  player-card: ', get_card_info(pc))
    
    if status == 'bust':
        for c in dc: update_card_counter(get_card_info(c)) # ★追加
        soc.close()
        reward = player.update_money(rate=rate)
        return reward, True, status
    return 0, False, status

def stand():
    print('Action: STAND')
    player.send_message(soc, 'stand')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for c in dc: update_card_counter(get_card_info(c)) # ★追加
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def double_down():
    print('Action: DOUBLE DOWN')
    player.double_bet()
    player.send_message(soc, 'double_down')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc)) # ★追加
    for c in dc: update_card_counter(get_card_info(c)) # ★追加
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def surrender():
    print('Action: SURRENDER')
    player.send_message(soc, 'surrender')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for c in dc: update_card_counter(get_card_info(c)) # ★追加
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def retry():
    print('Action: RETRY')
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    update_card_counter(get_card_info(pc)) # ★追加
    if status == 'bust':
        for c in dc: update_card_counter(get_card_info(c)) # ★追加
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

# === 状態取得 (スライドの「17次元入力」を実装) ===
def get_state():
    p_hand, d_hand = get_current_hands()
    
    # 1. 基本情報（正規化）
    score = p_hand.get_score() / 30.0 
    length = p_hand.length() / 10.0
    d_score = d_hand.get_score() / 30.0
    
    # 2. ソフトハンドフラグ (Aを11として使えるか)
    has_ace = False
    raw_score = 0
    for card_id in p_hand.cards:
        rank = (card_id % 13) + 1
        if rank == 1: has_ace = True
        raw_score += min(10, rank)
    soft_hand_val = 1.0 if (has_ace and raw_score + 10 <= 21) else 0.0

    # 3. カウンティング情報（確率に正規化）
    remaining = max(1, N_TOTAL_CARDS_INIT - g_total_cards_seen)
    norm_counter = g_card_counter.astype(np.float32) / remaining

    # 結合 (4 + 13 = 17次元)
    state_arr = np.concatenate([
        np.array([score, length, soft_hand_val, d_score]), 
        norm_counter
    ]).astype(np.float32)
    
    return state_arr

# === 行動選択 (DQN版) ===
def select_action(state, strategy: Strategy):
    global steps_done, EPS
    
    # テストモードまたは活用フェーズ
    if strategy == Strategy.QMAX or (strategy == Strategy.E_GREEDY and random.random() > EPS):
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(device)
            return ACTION_LIST[policy_net(t).argmax().item()]
    
    # 探索フェーズ (ランダム)
    return np.random.choice(ACTION_LIST)

# === 学習ステップ ===
def optimize_model():
    if len(memory) < BATCH_SIZE: return None
    
    transitions = random.sample(memory, BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch_done, dtype=torch.float32).to(device)

    # Q(s,a)
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # max Q(s',a') (Target Net)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

### メイン処理 ###
def main():
    global g_retry_counter, steps_done, EPS

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--history', type=str, default='play_log.csv')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--testmode', action='store_true')
    args = parser.parse_args()

    n_games = args.games + 1
    money_history = []
    loss_history = []
    
    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile)

    initialize_card_counter() # カウンター初期化

    for n in range(1, n_games):
        game_start(n)
        state = get_state()
        
        # εの線形減衰 (スライドの設定を反映)
        if not args.testmode:
            if n < EPS_DECAY:
                ratio = n / EPS_DECAY
                EPS = EPS_START - (EPS_START - EPS_END) * ratio
            else:
                EPS = EPS_END
        else:
            EPS = 0.0 # テストモードは探索なし

        while True:
            if args.testmode:
                action = select_action(state, Strategy.QMAX)
            else:
                action = select_action(state, Strategy.E_GREEDY)
            
            # RETRY上限チェック
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                action = np.random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])

            reward, done, status = act(action)
            if action == Action.RETRY: g_retry_counter += 1

            # === ★最重要: 報酬の正規化 & ペナルティの導入 ===
            # スライドの「反省と教訓」にある、RETRY踏み倒しを防ぐためのロジック
            norm_reward = reward / player.basic_bet
            
            if action == Action.RETRY:
                norm_reward = -1.0 # RETRYには「負け」と同等の重いペナルティ
            elif status == 'bust':
                norm_reward = -1.0 # バーストも負け
            elif action == Action.SURRENDER:
                norm_reward = -0.5 # サレンダーは被害最小限

            next_state = get_state()

            # 学習 (Testmodeでなければ)
            if not args.testmode:
                action_idx = ACTION_LIST.index(action)
                memory.append((state, action_idx, norm_reward, next_state, done))
                
                loss = optimize_model()
                if loss is not None:
                    loss_history.append(loss)
                    steps_done += 1
                    if steps_done % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            print(f'{state[0]},{state[1]},{get_action_name(action)},{status},{reward}', file=logfile)
            state = next_state

            if done:
                money_history.append(player.get_money())
                break
        print('')

    logfile.close()
    
    # グラフ表示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(money_history)
    plt.title("Money History")
    plt.grid()
    
    if not args.testmode:
        plt.subplot(1, 2, 2)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.grid()
    plt.show()

    if args.save != '':
        torch.save(policy_net.state_dict(), args.save)
        print(f"Model saved to {args.save}")

if __name__ == '__main__':
    main()