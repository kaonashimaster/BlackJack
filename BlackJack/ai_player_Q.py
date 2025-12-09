import copy
import socket
import argparse
import numpy as np
import pickle
import os

# 相対インポート (python -m BlackJack.ai_player_Q で実行する場合)
from .classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from .config import PORT, BET, INITIAL_MONEY, N_DECKS

# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10

### グローバル変数 ###
g_retry_counter = 0
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)
soc = None
q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値
# 学習時はこの設定で動く
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPS_START = 1.0
EPS_END = 0.00  # 本番では0にするため
EPS_DECAY_GAMES = 40000 # 5万回のうち4万回かけてじっくりランダムを減らす

### 関数 ###

def game_start(game_ID=0):
    global g_retry_counter, player, soc
    g_retry_counter = 0
    bet, money = player.set_bet()
    # メッセージ送信はしない（プロトコル合わせ）

def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# Act関数：行動を送信し、結果を受信し、報酬を計算する
def act(action):
    global player, soc, g_retry_counter

    # 1. コマンド送信
    if action == Action.HIT: cmd = 'hit'
    elif action == Action.STAND: cmd = 'stand'
    elif action == Action.DOUBLE_DOWN: cmd = 'double_down'
    elif action == Action.SURRENDER: cmd = 'surrender'
    elif action == Action.RETRY: cmd = 'retry'
    else: cmd = 'stand'
    
    player.send_message(soc, cmd)

    # 2. 結果受信
    try:
        if action == Action.HIT or action == Action.DOUBLE_DOWN:
            pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
        elif action == Action.RETRY:
            pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
            g_retry_counter += 1
        elif action == Action.STAND or action == Action.SURRENDER:
            score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
        else:
            return 0.0, True, 'error', 0.0
    except Exception as e:
        return 0.0, True, 'error', 0.0

    # 3. 学習用報酬の計算（ここが必勝のキモ）
    learning_reward = 0.0
    
    # 基本の勝ち負け
    if status == 'win' or status == 'dealer_bust':
        learning_reward = 1.0
    elif status == 'lose':
        learning_reward = -1.0
    elif status == 'bust':
        learning_reward = -5.0  # ★バーストは死罪（絶対に避けるように学習させる）
    elif status == 'push' or status == 'draw':
        learning_reward = 0.0
    elif status == 'surrendered':
        learning_reward = -0.5
    
    # RETRYのコスト（少しだけ嫌がらせるが、バースト(-5.0)よりはマシと思わせる）
    if action == Action.RETRY:
        learning_reward -= 0.1

    # 実際の獲得金額
    final_reward = 0.0
    done = (status != 'unsettled')
    if done:
        final_reward = player.update_money(rate=rate)

    return final_reward, done, status, learning_reward

# ★重要：状態の定義を「ガチ勢」仕様に変更
def get_state():
    # 1. 自分の点数
    score = player.player_hand.get_score()
    
    # 2. ディーラーのカード（見えている1枚）
    if len(player.dealer_hand.cards) > 0:
        d_card = player.dealer_hand.cards[0]
        d_score = min(10, (d_card % 13) + 1)
    else:
        d_score = 0
        
    # 3. ソフトハンドかどうか（Aを11として持っているか）
    # これがないと「A-6(17)」と「10-7(17)」を区別できず負ける
    has_usable_ace = False
    hand_val = 0
    has_ace = False
    for c in player.player_hand.cards:
        rank = (c % 13) + 1
        if rank == 1: has_ace = True
        hand_val += min(10, rank)
    
    # 「Aを持っていて」かつ「Aを11とみなしてもバーストしない」ならソフトハンド
    if has_ace and (hand_val + 10 <= 21):
        has_usable_ace = True

    # 状態をタプルで返す（Qテーブルのキーになる）
    return (score, d_score, has_usable_ace)

# 行動選択（イプシロン・グリーディ）
def select_action(state, epsilon):
    # RETRY回数上限ならRETRY以外の行動から選ぶ
    available_actions = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]
    if g_retry_counter >= RETRY_MAX:
        available_actions.remove(Action.RETRY)

    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        # Q値が最大の行動を選ぶ
        return q_table.get_best_action(state, available_actions=available_actions)

### QTableクラスの拡張（特定の選択肢から選べるように） ###
# classes.py をいじりたくないので、簡易的にここでオーバーライド的な処理
def get_best_action_custom(q_table_obj, state, available_actions):
    best_actions = []
    best_value = -float('inf')
    
    for a in available_actions:
        q = q_table_obj.get_Q_value(state, a)
        if q > best_value:
            best_value = q
            best_actions = [a]
        elif q == best_value:
            best_actions.append(a)
    
    if not best_actions:
        return np.random.choice(available_actions)
    return np.random.choice(best_actions)

# QTableにメソッドを追加（モンキーパッチ）
QTable.get_best_action = get_best_action_custom


### メイン処理 ###
def main():
    global g_retry_counter, player, soc, q_table

    parser = argparse.ArgumentParser(description='Q-Learning AI Player')
    parser.add_argument('--games', type=int, default=1000, help='number of games')
    parser.add_argument('--load', type=str, default='', help='load Q-table from file')
    parser.add_argument('--save', type=str, default='', help='save Q-table to file')
    parser.add_argument('--train', action='store_true', help='training mode (high epsilon)')
    args = parser.parse_args()

    n_games = args.games
    
    # Qテーブル読み込み
    if args.load != '' and os.path.exists(args.load):
        q_table.load(args.load)
        print(f"Q-table loaded from {args.load}")

    # ソケット接続
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    total_wins = 0
    initial_money = player.get_money()

    for n in range(1, n_games + 1):
        game_start(n)
        
        # 1. シャッフル通知・初期カード受信
        shuffled = player.receive_card_shuffle_status(soc)
        player.receive_init_cards(soc)
        
        # 現在の状態取得
        state = get_state()
        
        # 学習率とランダム率の決定
        if args.train:
            # 学習モード：徐々にランダムを減らす
            epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (n / EPS_DECAY_GAMES))
            lr = LEARNING_RATE
        else:
            # 本番モード：ランダムなし（ガチ）
            epsilon = 0.0
            lr = 0.0 # 学習もしない

        while True:
            # 行動選択
            action = select_action(state, epsilon)
            
            # 行動実行
            reward, done, status, learning_reward = act(action)
            
            # 次の状態
            next_state = get_state()
            
            # Q値更新（学習モードのみ）
            if args.train:
                current_q = q_table.get_Q_value(state, action)
                # 次の状態での最大Q値
                max_next_q = 0
                if not done:
                    # 次もゲームが続くなら、その先の最大Q値を考慮
                    # RETRY上限ならRETRYを除外して最大値を探す
                    avail = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]
                    if g_retry_counter >= RETRY_MAX: avail.remove(Action.RETRY)
                    
                    # 簡易的に全アクションから最大を取得（厳密には上記availから）
                    vals = [q_table.get_Q_value(next_state, a) for a in avail]
                    max_next_q = max(vals) if vals else 0
                
                # Q学習の式
                new_q = current_q + lr * (learning_reward + DISCOUNT_FACTOR * max_next_q - current_q)
                q_table.set_Q_value(state, action, new_q)

            state = next_state
            
            if done:
                if status == 'win': total_wins += 1
                break
        
        # 途中経過
        if n % 1000 == 0:
            print(f"Game {n}: WinRate={total_wins/n*100:.2f}%, Money={player.get_money()}")

    # 終了処理
    print(f"\nFinal Money: {player.get_money()} (Profit: {player.get_money() - initial_money})")
    print(f"Final Win Rate: {total_wins/n_games*100:.2f}%")
    
    soc.close()

    # Qテーブル保存
    if args.save != '':
        q_table.save(args.save)
        print(f"Q-table saved to {args.save}")

if __name__ == '__main__':
    main()