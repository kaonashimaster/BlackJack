import torch
import torch.nn as nn
import torch.nn.functional as F

# 外部ライブラリ(mylib)への依存を削除し、PyTorch標準機能だけで記述
# これにより Dropout や Batch Normalization の混入を完全に防ぐ

class BJNet(nn.Module):

    def __init__(self):
        super(BJNet, self).__init__()

        # ★修正1: 入力を 17 -> 18 に変更 (リトライ回数を追加したため)
        # 1層目: 18次元入力 → 256ノード
        # Batch Normalization (do_bn) は削除
        self.layer1 = nn.Linear(18, 256)

        # 2層目: 256ノード → 256ノード
        # Batch Normalization と Dropout (0.3) は削除
        self.layer2 = nn.Linear(256, 256)

        # 3層目: 256ノード → 5クラス出力 (HIT, STAND, DD, SURRENDER, RETRY)
        self.layer3 = nn.Linear(256, 5)

    def forward(self, x):
        # 活性化関数 ReLU はそのまま採用
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        # 出力層は活性化関数なし (生のQ値を出力)
        y = self.layer3(h)
        return y