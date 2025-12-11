import torch
import copy
import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt  # 【追加】グラフ描画用
from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from NN_structure import BJNet

# === カウンティング用グローバル変数 ===
g_card_counter = np.zeros(13, dtype=int) 
g_total_cards_seen = 0

# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10


### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# Q学習用のQテーブル
q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値
EPS = 0.3 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率

# DQNモデルの読み込み
try:
    q_net = BJNet()
    target_model = BJNet()
except Exception as e:
    print("Model Init Error")

### 関数 ###

# ゲームを開始する
def game_start(game_ID=0):
    global g_retry_counter, player, soc

    print('Game {0} start.'.format(game_ID))
    print('  money: ', player.get_money(), '$')

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

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
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        print('Dealer said: Card set has been shuffled before this game.')
        initialize_card_counter()

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
    # カウンティングの更新
    update_card_counter(get_card_info(pc))
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
        #カウンティングの更新
        update_card_counter(get_card_info(dc[i]))
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
        #カウンティんぐの更新
        update_card_counter(get_card_info(dc[i]))
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
    #カウンティングの更新
    update_card_counter(get_card_info(dc[i]))
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
    
    #カウンティングの更新
    update_card_counter(get_card_info(pc))
    
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

# 現在の状態の取得
def get_state():

    # 現在の手札情報を取得
    p_hand, d_hand = get_current_hands()

    # 「現在の状態」を設定
    score = p_hand.get_score() # プレイヤー手札のスコア
    length = p_hand.length() # プレイヤー手札の枚数
    
    d_score = d_hand.get_score()

    has_ace = False
    raw_score_assuming_ace_is_1 = 0
    for card_id in p_hand.cards:
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
    
    # DQN用にテンソル形式で返す
    state_vector = np.concatenate([
        np.array([score, length, soft_hand_val, d_score]), 
        g_card_counter.astype(np.float32)
    ]).astype(np.float32)
    
    return torch.from_numpy(state_vector).unsqueeze(0)

# 行動戦略
def select_action(state, strategy: Strategy):
    global q_net

    # DQNモデルを使って行動を決定
    qvalues = q_net(state)
    action_index = torch.argmax(qvalues).item()

    action_mapping = {
        0: Action.HIT,
        1: Action.STAND,
        2: Action.DOUBLE_DOWN,
        3: Action.SURRENDER,
        4: Action.RETRY
    }

    return action_mapping[action_index]
            

### ここから処理開始 ###

def main():
    global g_retry_counter, player, soc, q_net

    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1, help='num. of games to play')
    parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    args = parser.parse_args()

    n_games = args.games + 1
    
    # 金額の推移記録用リスト
    money_history = [player.get_money()]

    # Q_networkの重みをロード
    if args.load != '':
        q_net.load_state_dict(torch.load(args.load))
        
    # テストモードなら推論に切り替え（エラー回避）
    if args.testmode:
        q_net.eval()

    # ログファイルを開く
    logfile = open(args.history, 'w')
    print('score,hand_length,action,result,reward', file=logfile)

    # n_games回ゲームを実行
    for n in range(1, n_games):

        # nゲーム目を開始
        game_start(n)

        # 「現在の状態」を取得
        state = get_state()

        while True:

            # 次に実行する行動を選択
            if args.testmode:
                action = select_action(state, Strategy.QMAX)
            else:
                action = select_action(state, Strategy.E_GREEDY)
            if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                # RETRY回数が上限に達しているにもかかわらずRETRYが選択された場合，他の行動をランダムに選択
                action = np.random.choice([
                    Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
                ])
            action_name = get_action_name(action) # 行動名を表す文字列を取得

            # 選択した行動を実際に実行
            reward, done, status = act(action)

            # 実行した行動がRETRYだった場合はRETRY回数カウンターを1増やす
            if action == Action.RETRY:
                g_retry_counter += 1

            # 「現在の状態」を再取得
            prev_state = state # 行動前の状態を別変数に退避
            state = get_state()

            # Qテーブルを更新
            if not args.testmode:
                # 学習処理（省略）
                pass

            # 【ここも復活しました！】ログファイルに記録（DQNのテンソル対応版）
            print('{},{},{},{},{}'.format(prev_state[0][0].item(), prev_state[0][1].item(), action_name, status, reward), file=logfile)

            # 終了フラグが立った場合はnゲーム目を終了
            if done == True:
                # 終了時の所持金を記録
                money_history.append(player.get_money())
                break

        print('')

    # ログファイルを閉じる
    logfile.close()

    # 【ここも復活しました！】所持金の推移をグラフで表示
    plt.figure(figsize=(10, 6))
    plt.plot(money_history, label='Money History')
    plt.axhline(y=INITIAL_MONEY, color='r', linestyle='--', label='Initial Money') # 初期所持金のライン
    plt.title(f'Money Trend over {args.games} Games')
    plt.xlabel('Games')
    plt.ylabel('Money ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Qテーブルをセーブ
    if args.save != '':
        # q_table.save(args.save)
        pass

if __name__ == '__main__':
    main()