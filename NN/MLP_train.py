import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from .networks import SampleMLP
from mylib.data_io import CSVBasedDataset
from mylib.utility import print_args
import matplotlib.pyplot as plt
import numpy as np


# データセットファイル
DATASET_CSV = './NN/csv_data/student_exam_scores.csv'

# 学習結果の保存先フォルダ
MODEL_DIR = './NN/MLP_models'

# --- 1回分の学習を実行する関数 ---
def train_single_run(DEVICE, N_EPOCHS, BATCH_SIZE, MODEL_DIR):
    # lossを記録するためのリストを初期化
    train_loss_list = []
    valid_loss_list = []
    
    # CSVファイルを読み込み, データセットを用意
    dataset = CSVBasedDataset(
        filename=DATASET_CSV,
        items=[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'], 'exam_score'],
        dtypes=[np.float32, np.float32]
    )

    # 訓練データセットと検証データセットに分割
    train_size = int(0.95 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル（ニューラルネットワーク）
    model = SampleMLP().to(DEVICE)
    
    # 最適化手法
    optimizer = optim.Adam(model.parameters())
    
    # 損失関数
    loss_func = nn.MSELoss()

    # 学習ループ
    for epoch in range(N_EPOCHS):
        # 訓練
        model.train()
        sum_train_loss = 0
        for X, Y in train_dataloader:
            X = X.to(DEVICE)
            Y = Y.float().view(-1, 1).to(DEVICE)
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = loss_func(Y_pred, Y)
            loss.backward()
            optimizer.step()
            sum_train_loss += float(loss.detach()) * len(X)
        avg_train_loss = sum_train_loss / train_size
        train_loss_list.append(avg_train_loss)

        # 検証
        model.eval()
        sum_valid_loss = 0
        with torch.inference_mode():
            for X, Y in valid_dataloader:
                X = X.to(DEVICE)
                Y = Y.float().view(-1, 1).to(DEVICE)
                Y_pred = model(X)
                loss = loss_func(Y_pred, Y)
                sum_valid_loss += float(loss.detach()) * len(X)
        avg_valid_loss = sum_valid_loss / valid_size
        valid_loss_list.append(avg_valid_loss)

    # この関数は、最後にlossのリストを返す
    return train_loss_list, valid_loss_list

# --- 全体を管理するメイン関数 ---
def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Multi-Layer Perceptron Sample Code (training)')
    parser.add_argument('--epochs', '-e', default=50, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=50, type=int, help='minibatch size')
    args = print_args(parser.parse_args())
    DEVICE = 'cpu' # CPUを使用
    N_EPOCHS = args['epochs']
    BATCH_SIZE = args['batchsize']

    N_RUNS = 10  # 実行回数を10回に設定
    all_train_losses = []
    all_valid_losses = []

    # 10回学習を繰り返すループ
    for i in range(N_RUNS):
        print(f'--- Run {i+1}/{N_RUNS} ---')
        # 1回分の学習を実行し、結果を受け取る
        train_loss, valid_loss = train_single_run(DEVICE, N_EPOCHS, BATCH_SIZE, MODEL_DIR)
        all_train_losses.append(train_loss)
        all_valid_losses.append(valid_loss)
    
    # --- 平均と標準偏差を計算 ---
    train_losses_np = np.array(all_train_losses)
    valid_losses_np = np.array(all_valid_losses)
    
    mean_train_loss = np.mean(train_losses_np, axis=0)
    std_train_loss = np.std(train_losses_np, axis=0)
    mean_valid_loss = np.mean(valid_losses_np, axis=0)
    std_valid_loss = np.std(valid_losses_np, axis=0)

    # --- 平均学習曲線のグラフを描画 ---
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, N_EPOCHS + 1)
    
    plt.plot(epochs_range, mean_train_loss, label='mean_train_loss')
    plt.plot(epochs_range, mean_valid_loss, label='mean_valid_loss')
    
    plt.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2)
    plt.fill_between(epochs_range, mean_valid_loss - std_valid_loss, mean_valid_loss + std_valid_loss, alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Average Loss Curve over {N_RUNS} runs\nBatchSize: {BATCH_SIZE}')
    plt.legend()
    plt.grid()
    
    graph_filename = os.path.join(MODEL_DIR, f'average_loss_layer4_epoch{N_EPOCHS}_batchsize{BATCH_SIZE}_percep50-100-50.png')
    plt.savefig(graph_filename)
    print(f'Average loss curve graph saved to {graph_filename}')


if __name__ == '__main__':
    main()