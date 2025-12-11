import argparse
import numpy as np
import gymnasium as gym
from classes import Strategy, QTable
from enum import Enum
import matplotlib.pyplot as plt # グラフ描画ライブラリを追加

# --- 基本設定 (変更なし) ---
GAME_NAME = 'CartPole-v1'

class Action(Enum):
    GO_LEFT = 0
    GO_RIGHT = 1
    UNDEFINED = 2
ACTION_ID = { Action.GO_LEFT:0, Action.GO_RIGHT:1 }
N_ACTIONS = len(ACTION_ID)

def get_action_name(action: Action):
    if action == Action.GO_LEFT:
        return 'GO_LEFT'
    elif action == Action.GO_RIGHT:
        return 'GO_RIGHT'
    else:
        return 'UNDEFINED'

# --- Q学習の固定パラメータ ---
LEARNING_RATE = 0.1   # 学習率 (今回は固定)
DISCOUNT_FACTOR = 0.9 # 割引率 (今回は固定)

# --- 状態の定義 (変更なし) ---
def get_state(observation):
    cart_pos = np.digitize(observation[0], bins=[-4.8, -2.4, 0, 2.4, 4.8])
    cart_vel = np.digitize(observation[1], bins=[-3.0, -1.5, 0, 1.5, 3.0])
    pole_ang = np.digitize(observation[2], bins=[-0.4, -0.2, 0, 0.2, 0.4])
    pole_vel = np.digitize(observation[3], bins=[-2.0, -1.0, 0, 1.0, 2.0])
    return (cart_pos, cart_vel, pole_ang, pole_vel)

# --- 行動選択関数 (EPSを引数で受け取るように変更) ---
def select_action(state, q_table, strategy: Strategy, current_eps):
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < current_eps: # グローバル変数EPSではなく引数を使う
            return np.random.choice([Action.GO_LEFT, Action.GO_RIGHT])
        else:
            return q_table.get_best_action(state)
    else: # RANDOM
        return np.random.choice([Action.GO_LEFT, Action.GO_RIGHT])

# --- 移動平均を計算するヘルパー関数 ---
def moving_average(data, window_size):
    """Calculate moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- ここから処理開始 ---
def main():
    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v1 Q-Learning Epsilon Experiment')
    parser.add_argument('--games', type=int, default=5000, help='num. of games to play for each epsilon') # デフォルトゲーム数を増やす
    parser.add_argument('--max_steps', type=int, default=500, help='max num. of steps per game') # CartPole-v1の最大ステップ数に合わせる
    # '--save' と '--load' は今回の実験では使わないので省略可
    # '--testmode' と '--randmode' も使わない
    args = parser.parse_args()

    N_GAMES = args.games
    MAX_STEPS = args.max_steps

    # --- 実験するEPSの値のリスト ---
    epsilon_values = [0.01, 0.1, 0.5, 0.9]
    results = {} # 各EPSの結果を保存する辞書

    # ゲーム環境を作成 (描画なしで高速化)
    env = gym.make(GAME_NAME)

    # --- 各EPS値で学習を実行 ---
    for eps_value in epsilon_values:
        print(f"\n--- Starting training for EPS = {eps_value} ---")
        
        # 新しいQテーブルを初期化
        q_table = QTable(action_class=Action, default_value=0)
        
        rewards_per_game = [] # このEPSでの各ゲームのリワードを記録

        # N_GAMES回、ゲームをプレイするループ
        for game_ID in range(1, N_GAMES + 1):
            observation, info = env.reset()
            state = get_state(observation)
            total_reward = 0

            # 1ゲーム内のループ
            for t in range(MAX_STEPS):
                # ε-greedyで行動を選択 (現在のeps_valueを渡す)
                action = select_action(state, q_table, Strategy.E_GREEDY, eps_value)

                prev_state = state

                observation, reward, done, truncated, info = env.step(ACTION_ID[action])
                state = get_state(observation)
                total_reward += reward

                # Qテーブルを更新
                _, V = q_table.get_best_action(state, with_value=True)
                Q = q_table.get_Q_value(prev_state, action)
                Q = (1 - LEARNING_RATE) * Q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * V)
                q_table.set_Q_value(prev_state, action, Q)

                if done or truncated: # truncatedも考慮
                    break
            
            rewards_per_game.append(total_reward)

            # 進捗表示 (例: 500ゲームごと)
            if game_ID % 500 == 0:
                avg_reward_last_100 = np.mean(rewards_per_game[-100:]) if len(rewards_per_game) >= 100 else np.mean(rewards_per_game)
                print(f"Game {game_ID}/{N_GAMES}, Average Reward (Last 100): {avg_reward_last_100:.2f}")

        # このEPSの結果を保存
        results[eps_value] = rewards_per_game
        print(f"--- Finished training for EPS = {eps_value} ---")

    env.close()

    # --- 結果をグラフで表示 ---
    plt.figure(figsize=(12, 6))
    window_size = 100 # 移動平均のウィンドウサイズ

    for eps_value, rewards in results.items():
        # 移動平均を計算
        avg_rewards = moving_average(rewards, window_size)
        # 移動平均はデータ数が減るので、対応するゲーム数を計算
        games_for_avg = np.arange(window_size, N_GAMES + 1)
        plt.plot(games_for_avg, avg_rewards, label=f'EPS = {eps_value}')

    plt.xlabel("Games")
    plt.ylabel(f"Average Reward (Moving Average over {window_size} games)")
    plt.title(f"CartPole Q-Learning Performance (LR={LEARNING_RATE}, DF={DISCOUNT_FACTOR})")
    plt.legend()
    plt.grid(True)
    plt.savefig("cartpole_epsilon_comparison.png") # グラフをファイルに保存
    print("\nGraph saved as cartpole_epsilon_comparison.png")
    plt.show() # グラフを表示


if __name__ == '__main__':
    main()
