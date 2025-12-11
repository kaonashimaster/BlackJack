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
# (EPSとLEARNING_RATEは動的に変更するため、初期値を設定)
EPS_START = 1.0         # 探索率の初期値
EPS_END = 0.01          # 探索率の最終値
EPS_DECAY_GAMES = 1000  # 探索率を何ゲームかけて減衰させるか

LR_START = 0.1          # 学習率の初期値
LR_END = 0.01           # 学習率の最終値
LR_DECAY_GAMES = 4000   # 学習率を何ゲームかけて減衰させるか

DISCOUNT_FACTOR = 0.9   # 割引率 (固定)

q_table = QTable(action_class=Action, default_value=0)

# --- 行動選択関数 (EPSを引数で受け取る) ---
def select_action(state, strategy: Strategy, current_eps):
    global q_table
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < current_eps: # 引数のcurrent_epsを使う
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
    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v1 Q-Learning with Decay')
    parser.add_argument('--games', type=int, default=5000, help='num. of games to play')
    parser.add_argument('--max_steps', type=int, default=500, help='max num. of steps per game')
    args = parser.parse_args()

    N_GAMES = args.games
    MAX_STEPS = args.max_steps

    env = gym.make(GAME_NAME)
    
    rewards_per_game = [] # 各ゲームのリワードを記録
    epsilon_history = [] # EPSの推移を記録
    lr_history = [] # 学習率の推移を記録

    # --- N_GAMES回、ゲームをプレイするループ ---
    for game_ID in range(1, N_GAMES + 1):

        # --- パラメータの減衰（Decay）計算 ---
        # EPSを線形に減衰させる (1.0から0.01へ)
        current_eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (game_ID / EPS_DECAY_GAMES))
        
        # 学習率を線形に減衰させる (0.1から0.01へ)
        current_lr = max(LR_END, LR_START - (LR_START - LR_END) * (game_ID / LR_DECAY_GAMES))
        
        epsilon_history.append(current_eps)
        lr_history.append(current_lr)
        # --- ここまでが減衰計算 ---

        observation, info = env.reset()
        state = get_state(observation)
        total_reward = 0

        # 1ゲーム内のループ
        for t in range(MAX_STEPS):
            # ε-greedyで行動を選択 (計算した current_eps を渡す)
            action = select_action(state, Strategy.E_GREEDY, current_eps)
            prev_state = state

            observation, reward, done, truncated, info = env.step(ACTION_ID[action])
            state = get_state(observation)
            total_reward += reward

            # Qテーブルを更新 (計算した current_lr を使う)
            _, V = q_table.get_best_action(state, with_value=True)
            Q = q_table.get_Q_value(prev_state, action)
            Q = (1 - current_lr) * Q + current_lr * (reward + DISCOUNT_FACTOR * V) # current_lrを使用
            q_table.set_Q_value(prev_state, action, Q)

            if done or truncated:
                break
        
        rewards_per_game.append(total_reward)

        if game_ID % 500 == 0:
            avg_reward_last_100 = np.mean(rewards_per_game[-100:])
            print(f"Game {game_ID}/{N_GAMES}, Avg Reward(100): {avg_reward_last_100:.2f}, EPS: {current_eps:.3f}, LR: {current_lr:.3f}")

    env.close()

    # --- 結果をグラフで表示 ---
    # 2つのグラフを縦に並べて表示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # グラフ1: 報酬の推移
    window_size = 100
    avg_rewards = moving_average(rewards_per_game, window_size)
    games_for_avg = np.arange(window_size, N_GAMES + 1)
    ax1.plot(games_for_avg, avg_rewards, label='Average Reward (100 games)', color='blue')
    ax1.set_ylabel("Average Reward")
    ax1.set_title(f"CartPole Q-Learning Performance (DF={DISCOUNT_FACTOR},EPSDENCY={EPS_DECAY_GAMES},LRDECAY={LR_DECAY_GAMES})")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # グラフ2: EPSと学習率の推移
    ax2.plot(range(1, N_GAMES + 1), epsilon_history, label='Epsilon (EPS)', color='green')
    ax2.plot(range(1, N_GAMES + 1), lr_history, label='Learning Rate (LR)', color='red')
    ax2.set_xlabel("Games")
    ax2.set_ylabel("Parameter Value")
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig("cartpole_decay_performance.png")
    print("\nGraph saved as cartpole_decay_performance.png")
    plt.show()

if __name__ == '__main__':
    main()