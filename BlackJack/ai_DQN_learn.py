import torch
import torch.nn as nn
import torch.optim as optim
import copy
import socket
import argparse
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt

# 相対インポート対策
try:
    from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
    from config import PORT, BET, INITIAL_MONEY, N_DECKS
    from NN_structure import BJNet
except ImportError:
    from .classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
    from .config import PORT, BET, INITIAL_MONEY, N_DECKS
    from .NN_structure import BJNet

# === カウンティング用グローバル変数 ===
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10

### グローバル変数 ###

g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None

# Q学習用のQテーブル (残しておく)
q_table = QTable(action_class=Action, default_value=0)

# === DQNハイパーパラメータ ===
EPS_START = 1.0       
EPS_END = 0.01        
EPS_DECAY = 40000     
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000 
TARGET_UPDATE_FREQ = 100    

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQNモデルの読み込み
try:
    q_net = BJNet().to(device)
    target_model = BJNet().to(device)
    target_model.load_state_dict(q_net.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    
except Exception as e:
    print(f"Model Init Error: {e}")

### 関数 ###

def game_start(game_ID=0):
    global g_retry_counter, player, soc
    g_retry_counter = 0

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        soc.connect((socket.gethostname(), PORT))
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except:
        pass

    bet, money = player.set_bet()
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        initialize_card_counter()

    dc, pc1, pc2 = player.receive_init_cards(soc)
    update_card_counter(get_card_info(dc))
    update_card_counter(get_card_info(pc1))
    update_card_counter(get_card_info(pc2))

def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# --- 行動関数群 ---
def hit():
    global player, soc
    player.send_message(soc, 'hit')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc))
    if status == 'bust':
        for i in range(len(dc)): update_card_counter(get_card_info(dc[i]))
        soc.close()
        reward = player.update_money(rate=rate)
        return reward, True, status
    else:
        return 0, False, status

def stand():
    global player, soc
    player.send_message(soc, 'stand')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for i in range(len(dc)): update_card_counter(get_card_info(dc[i]))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def double_down():
    global player, soc
    bet, money = player.double_bet()
    player.send_message(soc, 'double_down')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    update_card_counter(get_card_info(pc))
    for i in range(len(dc)): update_card_counter(get_card_info(dc[i]))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def surrender():
    global player, soc
    player.send_message(soc, 'surrender')
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    for i in range(len(dc)): update_card_counter(get_card_info(dc[i]))
    soc.close()
    reward = player.update_money(rate=rate)
    return reward, True, status

def retry():
    global player, soc
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    update_card_counter(get_card_info(pc))

    learning_reward = -penalty

    if status == 'bust':
        for i in range(len(dc)): update_card_counter(get_card_info(dc[i]))
        soc.close()
        reward = player.update_money(rate=rate)
        return reward + learning_reward, True, status
    else:
        return learning_reward, False, status

def act(action: Action):
    if action == Action.HIT: return hit()
    elif action == Action.STAND: return stand()
    elif action == Action.DOUBLE_DOWN: return double_down()
    elif action == Action.SURRENDER: return surrender()
    elif action == Action.RETRY: return retry()
    else: exit()

# --- カウンティングと状態取得 ---
def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_id_str):
    global g_card_counter, g_total_cards_seen
    if card_id_str == 'X' or card_id_str is None: return
    try:
        if isinstance(card_id_str, int): rank_idx = (card_id_str % 13)
        elif card_id_str.isdigit(): rank_idx = (int(card_id_str) % 13)
        else:
            parts = card_id_str.split('-')
            if len(parts) < 2: return
            rank_str = parts[1]
            if rank_str == 'A': rank_idx = 0
            elif rank_str == 'J': rank_idx = 10
            elif rank_str == 'Q': rank_idx = 11
            elif rank_str == 'K': rank_idx = 12
            else: rank_idx = int(rank_str) - 1
        if g_card_counter[rank_idx] > 0: 
            g_card_counter[rank_idx] -= 1
            g_total_cards_seen += 1
    except: pass

def get_state():
    p_hand, d_hand = get_current_hands()
    score = p_hand.get_score()
    length = p_hand.length()
    d_score = d_hand.get_score()
    has_ace = False
    raw = 0
    for cid in p_hand.cards:
        r = (cid % 13) + 1
        if r == 1: has_ace = True
        raw += min(10, r)
    soft = 1.0 if (has_ace and raw + 10 <= 21) else 0.0
    
    state_vector = np.concatenate([
        np.array([score, length, soft, d_score]), 
        g_card_counter.astype(np.float32)
    ]).astype(np.float32)
    
    return torch.from_numpy(state_vector).unsqueeze(0).to(device)

# --- 【修正】行動選択関数 ---
def select_action(state, strategy, epsilon=0.0):
    global q_net

    if strategy == Strategy.E_GREEDY and random.random() < epsilon:
        action = np.random.choice([
            Action.DOUBLE_DOWN, Action.HIT, Action.RETRY, Action.STAND, Action.SURRENDER
        ])
    else:
        # NNによる推論
        # ★【重要修正】推論モード(eval)に切り替える
        q_net.eval() 
        with torch.no_grad():
            qvalues = q_net(state)
            action_index = torch.argmax(qvalues).item()
            
            mapping = {0: Action.DOUBLE_DOWN, 1: Action.HIT, 2: Action.RETRY, 3: Action.STAND, 4: Action.SURRENDER}
            action = mapping[action_index]

    if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
        action = np.random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])

    return action

# --- 【修正】学習関数 ---
def train_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    # ★【重要修正】学習モード(train)に切り替える
    q_net.train()

    batch = random.sample(replay_buffer, BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.cat(state_batch)
    action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float32).unsqueeze(1)
    next_state_batch = torch.cat(next_state_batch)
    done_batch = torch.tensor(done_batch, device=device, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_q_values = target_model(next_state_batch).max(1)[0].unsqueeze(1)
    
    expected_q_values = reward_batch + (DISCOUNT_FACTOR * next_q_values * (1 - done_batch))

    loss = criterion(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### メイン処理 ###

def main():
    global g_retry_counter, player, soc, q_net

    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1, help='num. of games to play')
    parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
    parser.add_argument('--load', type=str, default='', help='filename to load')
    parser.add_argument('--save', type=str, default='', help='filename to save')
    parser.add_argument('--testmode', help='runs without learning', action='store_true')
    args = parser.parse_args()

    if args.load != '':
        if os.path.exists(args.load):
            q_net.load_state_dict(torch.load(args.load, map_location=device))
            target_model.load_state_dict(q_net.state_dict())
            print(f"Loaded model: {args.load}")

    if args.testmode:
        q_net.eval()

    action_to_idx = {Action.DOUBLE_DOWN:0, Action.HIT:1, Action.RETRY:2, Action.STAND:3, Action.SURRENDER:4}

    total_wins = 0
    money_history = [player.get_money()]

    print(f"Start processing {args.games} games...")

    for n in range(1, args.games + 1):
        
        # εの計算 (4万回で線形に0.01まで下がる)
        epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (n / EPS_DECAY))
        if args.testmode: epsilon = 0.0

        game_start(n)
        state = get_state()

        while True:
            if args.testmode:
                action = select_action(state, Strategy.QMAX, epsilon=0)
            else:
                action = select_action(state, Strategy.E_GREEDY, epsilon=epsilon)
            
            action_name = get_action_name(action)

            reward, done, status = act(action)
            if action == Action.RETRY:
                g_retry_counter += 1

            prev_state = state
            state = get_state() 

            if not args.testmode:
                act_idx = action_to_idx[action]
                done_val = 1.0 if done else 0.0
                replay_buffer.append((prev_state, act_idx, reward, state, done_val))
                train_model()

            if done:
                money_history.append(player.get_money())
                if status == 'win' or status == 'dealer_bust':
                    total_wins += 1
                break

        if not args.testmode and n % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(q_net.state_dict())

        if n % 100 == 0:
            print(f"Game {n}: Money={player.get_money()}, WinRate={(total_wins/n)*100:.1f}%, Eps={epsilon:.3f}")

    if args.save != '':
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save(q_net.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    print("\n" + "="*30)
    print(f"Total Games: {args.games}")
    print(f"Total Wins: {total_wins}")
    print(f"Win Rate: {(total_wins / args.games) * 100:.2f}%")
    print("="*30 + "\n")

    if not args.testmode and len(money_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(money_history)), money_history)
        plt.title(f'Money Trend over {args.games} Games')
        plt.grid(True)
        try:
            plt.show()
        except: pass

if __name__ == '__main__':
    main()