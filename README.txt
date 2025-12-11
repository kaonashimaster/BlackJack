ブラックジャックAI 提出課題
学籍番号: XXXXXXX
氏名: 〇〇 〇〇

【提出ファイルの内容】
- ソースコード一式 (BlackJack/, mylib/)
- 学習済みモデル (BJNet_models_DQN/dqn_model.pth)

【実行方法】
Python 3.12 環境で動作確認済みです。
必要なライブラリ: torch, numpy, matplotlib

1. ディーラーの起動
   cd BlackJack
   python dealer.py

2. AIの実行（テストモード）
   # ルートディレクトリから実行してください
   python -m BlackJack.ai_Deep_QNetwork --games 1000 --testmode

【アピールポイント】
- Deep Q-Network (DQN) によるオンライン学習を実装しました。
- カードカウンティング情報を確率化して入力することで勝率を高めました。