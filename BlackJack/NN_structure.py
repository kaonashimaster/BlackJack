import os
import sys
# sys.path.append(os.path.abspath('..')) # <- 不要
import torch
import torch.nn as nn
import torch.nn.functional as F

from mylib.basic_layers import *


# 「プレイヤースコア」「手札の枚数」の2情報から4種類の行動の選択確率を計算するニューラルネットワーク
class BJNet(nn.Module):

    def __init__(self):
        super(BJNet, self).__init__()

        # 1層目: 17次元入力 → 64パーセプトロン
        self.layer1 = FC(in_features=17, out_features=256, do_bn=True, activation='relu')

        # 2層目: 64パーセプトロン → 64パーセプトロン
        self.layer2 = FC(in_features=256, out_features=256, do_bn=True, dropout_ratio=0.3, activation='relu')

        # 3層目: 64パーセプトロン → 5クラス出力 (行動のQ値を予測)
        self.layer3 = FC(in_features=256, out_features=5, do_bn=False, activation='none')

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        y = self.layer3(h)
        return y