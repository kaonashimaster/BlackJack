import argparse
import numpy as np
import gymnasium as gym
from classes import Strategy, QTable
from enum import Enum
import matplotlib.pyplot as plt

# --- 基本設定 (変更なし) ---
GAME_NAME = 'CartPole-v1'

class Action(Enum):
    GO_LEFT = 0
    GO_RIGHT = 1
    UNDEFINED = 2
ACTION_ID = { Action.GO_LEFT:0, Action.GO_RIGHT:1 }
N_ACTIONS = len(ACTION_ID)

# (get_action_name, get_state関数は元のファイルから変更なし)
def get_action_name(action: Action):
    if action == Action.GO_LEFT:
        return 'GO_LEFT'
    elif action == Action.GO_RIGHT:
        return 'GO_RIGHT'
    else:
        return 'UNDEFINED'

def get_state(observation):
    cart_pos = np.digitize(observation[0], bins=[-4.8, -2.4, 0, 2.4, 4.8])
    cart_vel = np.digitize(observation[1], bins=[-3.0, -1.5, 0, 1.5, 3.0])
    pole_ang = np.digitize(observation[2], bins=[-0.4, -0.2, 0, 0.2, 0.4])
    pole_vel = np.digitize(observation[3], bins=[-2.0, -1.0, 0, 1.0, 2.0])
    return (cart_pos, cart_vel, pole_ang, pole_vel)

# --- Q学習の固定パラメータ ---
EPS = 0.1             # 探索率 (今回は固定)
DISCOUNT_FACTOR = 0.9 # 割引率 (今回は固定)

# --- 行動選択関数 (変更なし) ---
def select_action(state, q_table, strategy: Strategy, current_eps=EPS):
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < current_eps:
            return np.random.choice([Action.GO_LEFT, Action.GO_RIGHT])
        else:
            return q_table.get_best_action(state)
    else: # RANDOM
        return np.random.choice([Action.GO_LEFT, Action.GO_RIGHT])

# --- 移動平均を計算するヘルパー関数 ---
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- ここから処理開始 ---
def main():
    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v1 Q-Learning Learning Rate Experiment')
    parser.add_argument('--games', type=int, default=5000, help='num. of games to play for each LR')
    parser.add_argument('--max_steps', type=int, default=500, help='max num. of steps per game')
    args = parser.parse_args()

    N_GAMES = args.games
    MAX_STEPS = args.max_steps

    # --- 実験するLEARNING_RATEの値のリスト ---
    lr_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {} 

    env = gym.make(GAME_NAME)

    # --- 各LEARNING_RATE値で学習を実行 ---
    for lr_value in lr_values:
        print(f"\n--- Starting training for LEARNING_RATE = {lr_value} ---")
        
        q_table = QTable(action_class=Action, default_value=0)
        rewards_per_game = [] 

        for game_ID in range(1, N_GAMES + 1):
            observation, info = env.reset()
            state = get_state(observation)
            total_reward = 0

            for t in range(MAX_STEPS):
                action = select_action(state, q_table, Strategy.E_GREEDY, EPS)
                prev_state = state
                observation, reward, done, truncated, info = env.step(ACTION_ID[action])
                state = get_state(observation)
                total_reward += reward

                # Qテーブルを更新 (現在のlr_valueを使用)
                _, V = q_table.get_best_action(state, with_value=True)
                Q = q_table.get_Q_value(prev_state, action)
                Q = (1 - lr_value) * Q + lr_value * (reward + DISCOUNT_FACTOR * V) # lr_valueを使用
                q_table.set_Q_value(prev_state, action, Q)

                if done or truncated:
                    break
            
            rewards_per_game.append(total_reward)

            if game_ID % 500 == 0:
                avg_reward_last_100 = np.mean(rewards_per_game[-100:]) if len(rewards_per_game) >= 100 else np.mean(rewards_per_game)
                print(f"Game {game_ID}/{N_GAMES}, Average Reward (Last 100): {avg_reward_last_100:.2f}")

        results[lr_value] = rewards_per_game
        print(f"--- Finished training for LEARNING_RATE = {lr_value} ---")

    env.close()

    # --- 結果をグラフで表示 ---
    plt.figure(figsize=(12, 6))
    window_size = 100 

    for lr_value, rewards in results.items():
        avg_rewards = moving_average(rewards, window_size)
        games_for_avg = np.arange(window_size, N_GAMES + 1)
        plt.plot(games_for_avg, avg_rewards, label=f'LR = {lr_value}')

    plt.xlabel("Games")
    plt.ylabel(f"Average Reward (Moving Average over {window_size} games)")
    plt.title(f"CartPole Q-Learning Performance (EPS={EPS}, DF={DISCOUNT_FACTOR})")
    plt.legend()
    plt.grid(True)
    plt.savefig("cartpole_lr_comparison.png")
    print("\nGraph saved as cartpole_lr_comparison.png")
    plt.show()


if __name__ == '__main__':
    main()