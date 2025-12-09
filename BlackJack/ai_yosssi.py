import copy
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from DQN_structure import QNetwork
from collections import deque
import random


# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10


### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0
# 直前に実行した行動（状態特徴用）
g_last_action = Action.UNDEFINED

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# # Q学習用のQテーブル
# q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値
EPS = 0.2 # ε-greedyにおけるε
LEARNING_RATE = 0.01 # 学習率
DISCOUNT_FACTOR = 0.5 # 割引率


# ActionとNNの出力インデックスの対応表
action_list = [
    Action.HIT,
    Action.STAND,
    Action.DOUBLE_DOWN,
    Action.SURRENDER,
    Action.RETRY
]

def action_to_index(action: Action) -> int:
    return action_list.index(action)

def encode_last_action(action: Action) -> int:
    if action in action_list:
        return action_to_index(action)
    return len(action_list)

def card_value_from_id(card_id: int) -> int:
    if card_id is None:
        return 0
    return min(10, (card_id % 13) + 1)



### 関数 ###

# ゲームを開始する
def game_start(game_ID=0):
    global g_retry_counter, player, soc, g_last_action

    print('Game {0} start.'.format(game_ID))
    print('  money: ', player.get_money(), '$')

    # RETRY回数カウンターと直前行動の初期化
    g_retry_counter = 0
    g_last_action = Action.UNDEFINED

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット
    bet, money = player.set_bet()
    print('Action: BET')
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    # シャッフルが行われた場合は True が, 行われなかった場合は False が，変数 cardset_shuffled にセットされる
    # なお，本サンプルコードではここで取得した情報は使用していない
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        print('Dealer said: Card set has been shuffled before this game.')

    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    print('Delaer gave cards.')
    print('  dealer-card: ', get_card_info(dc))
    print('  player-card 1: ', get_card_info(pc1))
    print('  player-card 2: ', get_card_info(pc2))
    print('  current score: ', player.get_score())

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit():
    global player, soc

    print('Action: HIT')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand():
    global player, soc

    print('Action: STAND')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down():
    global player, soc

    print('Action: DOUBLE DOWN')

    # 今回のみベットを倍にする
    bet, money = player.double_bet()
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# SURRENDERを実行する
def surrender():
    global player, soc

    print('Action: SURRENDER')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# RETRYを実行する
def retry():
    global player, soc

    print('Action: RETRY')

    # ベット額の 1/4 を消費
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    print('  player-card {0} has been removed.'.format(player.get_num_player_cards()))
    print('  money: ', player.get_money(), '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'retry')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
        return reward-penalty, True, status

    # バーストしなかった場合は続行
    else:
        return -penalty, False, status

# 行動の実行
def act(action: Action):
    global g_last_action
    g_last_action = action
    if action == Action.HIT:
        return hit()
    elif action == Action.STAND:
        return stand()
    elif action == Action.DOUBLE_DOWN:
        return double_down()
    elif action == Action.SURRENDER:
        return surrender()
    elif action == Action.RETRY:
        return retry()
    else:
        exit()


### これ以降の関数が重要 ###

# 現在の状態の取得
def get_state():

    # ??????????
    #   - p_hand: ???????
    #   - d_hand: ??????????????????
    p_hand, d_hand = get_current_hands()

    player_score = p_hand.get_score()
    dealer_upcard = card_value_from_id(d_hand.cards[0]) if d_hand.length() > 0 else 0
    usable_ace = int(p_hand.has_usable_ace())
    player_card_count = p_hand.length()
    dealer_card_count = d_hand.length()
    can_split = 0
    if player_card_count == 2:
        first = card_value_from_id(p_hand.cards[0])
        second = card_value_from_id(p_hand.cards[1])
        can_split = int(first == second)
    last_action_feature = encode_last_action(g_last_action)
    current_bet = player.get_current_bet()

    state = (
        player_score,
        dealer_upcard,
        usable_ace,
        can_split,
        player_card_count,
        dealer_card_count,
        last_action_feature,
        current_bet
    )

    return state

# 行動戦略
def select_action(state, strategy: Strategy):
    global q_table

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)

    # ε-greedy
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < EPS:
            return select_action(state, strategy=Strategy.RANDOM)
        else:
            return q_table.get_best_action(state)

    # ランダム戦略
    else:
        z = np.random.randint(0, 4)
        if z == 0:
            return Action.HIT
        elif z == 1:
            return Action.STAND
        elif z == 2:
            return Action.DOUBLE_DOWN
        elif z == 3:
            return Action.SURRENDER
        else: # z == 4 のとき
            return Action.RETRY
        
# DQNでの行動戦略
def select_action_dqn(state, strategy: Strategy):
    global q_net

    if np.random.rand() < EPS:
        # ランダムに行動を選択
        z = np.random.randint(0, 5)
        if z == 0:
            return Action.HIT
        elif z == 1:
            return Action.STAND
        elif z == 2:
            return Action.DOUBLE_DOWN
        elif z == 3:
            return Action.SURRENDER
        else: # z == 4 のとき
            return Action.RETRY
    
    state_tensor = torch.tensor([state], dtype=torch.float32)
    qvalues = q_net(state_tensor)
    action_index = torch.argmax(qvalues).item()

    action_mapping = {
        0: Action.HIT,
        1: Action.STAND,
        2: Action.DOUBLE_DOWN,
        3: Action.SURRENDER,
        4: Action.RETRY
    }

    return action_mapping[action_index]


# DQNの1ステップ学習
def train_step(q_net, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Q(s, a) の予測値（batch_size × action_dim）
    q_values = q_net(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q_target = r + γ max_a' Q(s', a')
    with torch.no_grad():
        next_q_values = q_net(next_states).max(1)[0]
        target = rewards + gamma * next_q_values * (1 - dones)

    loss = ((target - q_values)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)




### ここから処理開始 ###

def main():
    global g_retry_counter, player, soc, q_net, optimizer

    losses = []
    buffer = ReplayBuffer(capacity=20000)
    batch_size = 32


    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1, help='num. of games to play')
    parser.add_argument('--game_iters', type=int, default=100, help='num. of game iterations for learning')
    parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    args = parser.parse_args()

    n_games = args.games + 1

    # # Qテーブルをロード
    # if args.load != '':
    #     q_table.load(args.load)

    # ログファイルを開く
    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile) # ログファイルにヘッダ行（項目名の行）を出力

    q_net = QNetwork()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    # DQNのNNパラメータをロード
    if args.load != '':
        q_net.load_state_dict(torch.load(args.load))


    # n_games回ゲームを実行
    for n in range(1, n_games):

        # nゲーム目を開始
        game_start(n)

        # 「現在の状態」を取得
        state = get_state()

        while True:

            # 次に実行する行動を選択
            if args.testmode:
                action = select_action_dqn(state, Strategy.QMAX)
            else:
                action = select_action_dqn(state, Strategy.E_GREEDY)
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                # RETRY回数が上限に達しているにもかかわらずRETRYが選択された場合，他の行動をランダムに選択
                action = np.random.choice([
                    Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
                ])
            action_name = get_action_name(action) # 行動名を表す文字列を取得

            # 選択した行動を実際に実行
            # 戻り値:
            #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
            #   - reward: 獲得金額（ゲーム続行中の場合は 0 , ただし RETRY を実行した場合は1回につき -BET/4 ）
            #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
            reward, done, status = act(action)

            # 実行した行動がRETRYだった場合はRETRY回数カウンターを1増やす
            if action == Action.RETRY:
                g_retry_counter += 1

            # 「現在の状態」を再取得
            prev_state = state # 行動前の状態を別変数に退避
            prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
            state = get_state()
            score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

            buffer.push(prev_state, action_to_index(action), reward, state, done)

            # DQNで学習
            if not args.testmode:
                if len(buffer) > batch_size:
                    batch = buffer.sample(batch_size)
                    loss = train_step(q_net, optimizer, batch)
                    losses.append(loss)

            # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
            print('{},{},{},{},{}'.format(prev_state[0], prev_state[1], action_name, status, reward), file=logfile)

            # 終了フラグが立った場合はnゲーム目を終了
            if done == True:
                break

        print('')

    # ログファイルを閉じる
    logfile.close()

    # DQNのNNパラメータをセーブ
    if args.save != '':
        # DQNのNNパラメータをセーブ
        torch.save(q_net.state_dict(), args.save)

    # 損失のグラフを表示
    if not args.testmode:
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.show()

if __name__ == '__main__':
    main()