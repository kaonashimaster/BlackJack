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

from classes import Action, Strategy, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS

# === DQNモデル ===
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# RETRY上限
RETRY_MAX = 10

### グローバル変数 ###
g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None

# === カウンティング用 ===
N_TOTAL_CARDS_INIT = N_DECKS * 52
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

# === 学習パラメータ ===
BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 30000 
TARGET_UPDATE = 500
LEARNING_RATE = 0.0001
MEMORY_CAPACITY = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル構築
policy_net = SimpleDQN(17, 5).to(device)
target_net = SimpleDQN(17, 5).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_CAPACITY)
steps_done = 0

ACTION_LIST = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]

### カウンティング関数 ###
def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_info):
    global g_card_counter, g_total_cards_seen
    rank_idx = -1
    if isinstance(card_info, int):
        rank_idx = (card_info % 13)
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

### ゲーム進行関数 ###
def game_start(game_ID=0):
    global g_retry_counter, player, soc
    print('Game {0} start.'.format(game_ID))
    g_retry_counter = 0
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 学習中はベット額を固定（安定化のため）
    player.basic_bet = 20 
    
    bet, money = player.set_bet()
    print(f'Action: BET (money: {money}, bet: {bet})')

    if player.receive_card_shuffle_status(soc):
        print('Dealer said: Card set has been shuffled.')
        initialize_card_counter()

    dc, pc1, pc2 = player.receive_init_cards(soc)
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
    update_card_counter(get_card_info(pc))
    print('  player-card: ', get_card_info(pc))
    if status == 'bust':
        for c in dc: update_card_counter(get_card_info(c))
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
    # ペナルティ支払い
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    update_card_counter(get_card_info(pc))
    
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

# === 状態取得 ===
def get_state():
    p_hand, d_hand = get_current_hands()
    
    score = p_hand.get_score() / 30.0 
    length = p_hand.length() / 10.0
    d_score = d_hand.get_score() / 30.0
    
    has_ace = False
    raw_score = 0
    for card_id in p_hand.cards:
        rank = (card_id % 13) + 1
        if rank == 1: has_ace = True
        raw_score += min(10, rank)
    soft_hand_val = 1.0 if (has_ace and raw_score + 10 <= 21) else 0.0

    remaining = max(1, N_TOTAL_CARDS_INIT - g_total_cards_seen)
    norm_counter = g_card_counter.astype(np.float32) / remaining

    state_arr = np.concatenate([
        np.array([score, length, soft_hand_val, d_score]), 
        norm_counter
    ]).astype(np.float32)
    
    return state_arr

# === 行動選択 ===
def select_action(state, strategy: Strategy):
    global steps_done, EPS
    
    if strategy == Strategy.QMAX or (strategy == Strategy.E_GREEDY and random.random() > EPS):
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(device)
            return ACTION_LIST[policy_net(t).argmax().item()]
    
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

    # max Q(s',a')
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ★修正: 見やすい戦略レポート（STとSUを区別）
def print_strategy_table():
    print("\n=== AI Strategy Report (Hard Hand) ===")
    print("Dealer Up Card (Top) vs Player Score (Left)")
    print("   " + " ".join([f"{d:2}" for d in range(2, 12)]))
    
    # 略語マップ
    name_map = {
        'HIT': ' H', 'STAND': 'ST', 'DOUBLE_DOWN': 'DD', 'SURRENDER': 'SU', 'RETRY': 'RT'
    }

    with torch.no_grad():
        for p_score in range(12, 22):
            row_actions = []
            for d_score in range(2, 12): 
                base_state = np.array([p_score/30.0, 2/10.0, 0.0, d_score/30.0])
                zero_count = np.zeros(13, dtype=np.float32)
                state_arr = np.concatenate([base_state, zero_count]).astype(np.float32)
                
                t = torch.from_numpy(state_arr).unsqueeze(0).to(device)
                act_idx = policy_net(t).argmax().item()
                act_name = ACTION_LIST[act_idx].name
                row_actions.append(name_map.get(act_name, '??'))
            print(f"{p_score:2} " + "  ".join(row_actions))

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
    
    # 行動カウント用
    action_counts = {a:0 for a in ACTION_LIST}
    
    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile)

    initialize_card_counter()

    for n in range(1, n_games):
        game_start(n)
        state = get_state()
        
        if not args.testmode:
            if n < EPS_DECAY:
                ratio = n / EPS_DECAY
                EPS = EPS_START - (EPS_START - EPS_END) * ratio
            else:
                EPS = EPS_END
        else:
            EPS = 0.0

        while True:
            if args.testmode:
                action = select_action(state, Strategy.QMAX)
            else:
                action = select_action(state, Strategy.E_GREEDY)
            
            # 統計
            action_counts[action] += 1

            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                action = np.random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])

            reward, done, status = act(action)
            if action == Action.RETRY: g_retry_counter += 1

            # === ★重要修正: 強気なAIを作る報酬設定 ===
            # 勝ったら2倍ボーナス！
            norm_reward = reward / player.basic_bet
            if norm_reward > 0:
                norm_reward *= 2.0 
            
            # サレンダーは逃げ癖がつかないように少し厳しく(-0.7)
            if action == Action.SURRENDER:
                norm_reward = -0.7
            elif action == Action.RETRY:
                norm_reward = -0.5 # コスト相当
            elif status == 'bust':
                norm_reward = -1.0 

            next_state = get_state()

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
    
    # 統計表示
    if not args.testmode:
        print_strategy_table()
        print("\nAction Distribution:")
        total_acts = sum(action_counts.values())
        for a, c in action_counts.items():
            print(f"{a.name}: {c} ({c/total_acts:.1%})")

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