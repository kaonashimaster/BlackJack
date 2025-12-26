import torch
import torch.nn as nn
import copy
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

from NN_structure import BJNet
from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS

# === カウンティング用グローバル変数 ===
N_TOTAL_CARDS_INIT = N_DECKS * 52
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0
RETRY_MAX = 10

### グローバル変数 ###
g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None

# 設定値
EPS = 0.2 
LEARNING_RATE = 0.001 
DISCOUNT_FACTOR = 0.9 
BATCH_SIZE = 32
BUFFER_SIZE = 50000 

# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ★修正版: BJNetの「毒」を安全に抜く関数 ===
def sanitize_model(model):
    """
    BJNetに含まれるDropoutを切り、BatchNormを「何もしない層」に固定する
    """
    for m in model.modules():
        # Dropoutを無効化 (確率0%)
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        
        # BatchNormの無効化処理
        if isinstance(m, nn.BatchNorm1d):
            # ここで track_running_stats=False にするとバッチサイズ1で死ぬのでやらない。
            # 代わりに、パラメータを初期化して学習をロックする。
            m.eval() # この層だけは常にevalモード
            
            # 平均0, 分散1 に固定（入力そのまま出力）
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1.0)
            
            # スケール(gamma)とシフト(beta)も固定
            if m.weight is not None:
                m.weight.data.fill_(1.0)
                m.weight.requires_grad = False # 固定！
            if m.bias is not None:
                m.bias.data.zero_()
                m.bias.requires_grad = False # 固定！

try:
    q_net = BJNet().to(device)
    
    # 手術実行
    sanitize_model(q_net) 
    
    # モデル全体を常にevalモードにする
    # (DQNではBatchNormの学習は不要かつ有害なため)
    q_net.eval()
    
    # オプティマイザには、requires_grad=True のパラメータ（全結合層）だけが渡される
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, q_net.parameters()), lr=LEARNING_RATE)

except Exception as e:
    print(f"Model Init Error: {e}")
    q_net = None

# 行動リスト
ACTION_LIST = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]

def action_to_idx(action):
    try:
        return ACTION_LIST.index(action)
    except ValueError:
        return 0 

def idx_to_action(idx):
    return ACTION_LIST[idx]

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
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

# カウンティング関数
def initialize_card_counter():
    global g_card_counter, g_total_cards_seen
    initial_count = 4 * N_DECKS
    g_card_counter = np.full(13, initial_count, dtype=int) 
    g_total_cards_seen = 0

def update_card_counter(card_info):
    global g_card_counter, g_total_cards_seen
    if isinstance(card_info, int):
        rank_idx = (card_info % 13)
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
        except: return
    else: return
    if 0 <= rank_idx < 13 and g_card_counter[rank_idx] > 0:
        g_card_counter[rank_idx] -= 1
        g_total_cards_seen += 1

# ゲーム進行関数
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

# 状態取得
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
    remaining_cards = max(1, N_TOTAL_CARDS_INIT - g_total_cards_seen)
    normalized_counter = g_card_counter.astype(np.float32) / remaining_cards
    state_vector = np.concatenate([
        np.array([score, length, soft_hand_val, d_score]), 
        normalized_counter
    ]).astype(np.float32)
    return state_vector

# 行動選択
def select_action(state, testmode=False):
    global q_net, EPS
    if not testmode and random.random() < EPS:
        return np.random.choice(ACTION_LIST)
    
    # 常にevalモードで実行（BatchNorm固定）
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = q_net(state_tensor)
        action_idx = torch.argmax(q_values).item()
    
    return idx_to_action(action_idx)

# 学習ステップ
def train_step(batch):
    states, actions, rewards, next_states, dones = batch
    
    # evalモードのまま勾配計算（Dropout/BNなし、重み更新のみ）
    q_values = q_net(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = q_net(next_states).max(1)[0]
        target = rewards + DISCOUNT_FACTOR * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    global g_retry_counter, EPS

    buffer = ReplayBuffer(capacity=BUFFER_SIZE)
    loss_history = []
    money_history = []
    win_count = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--history', type=str, default='play_log.csv')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--testmode', action='store_true')
    args = parser.parse_args()
    n_games = args.games + 1
    
    if args.testmode: EPS = 0.0

    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile)
    initialize_card_counter()

    for n in range(1, n_games):
        game_start(n)
        state = get_state()

        while True:
            action = select_action(state, args.testmode)
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                action = np.random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])

            reward, done, status = act(action)
            
            # 報酬正規化
            normalized_reward = reward / player.basic_bet
            if action == Action.RETRY: normalized_reward = -1.0
            elif status == 'bust': normalized_reward = -1.0
            elif action == Action.SURRENDER: normalized_reward = -0.5

            if action == Action.RETRY: g_retry_counter += 1
            next_state = get_state()
            buffer.push(state, action_to_idx(action), normalized_reward, next_state, done)

            if not args.testmode and len(buffer) > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = train_step(batch)
                loss_history.append(loss)

            print(f'{state[0]*30:.0f},{state[1]*10:.0f},{get_action_name(action)},{status},{reward}', file=logfile)
            state = next_state

            if done:
                money_history.append(player.get_money())
                if status == 'win' or status == 'dealer_bust':
                    win_count += 1
                break
        print('')

    logfile.close()
    
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
        torch.save(q_net.state_dict(), args.save)
        print(f"Model saved to {args.save}")

if __name__ == '__main__':
    main()