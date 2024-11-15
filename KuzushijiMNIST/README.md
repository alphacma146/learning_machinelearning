# KuzushijiMNIST

:arrow_forward:[まずはここをチェック](Description/overview.md)

## フォルダ構成

```bash
[KuzushijiMNIST]
  ├ [.git]
  ├ [.github]
  ├ .gitignore
  ├ [Dataset]                           ...データフォルダ
  │  ├ [getter]
  │  │  └ kuzushi.py                    ...データ取得コード
  │  └ resources.zip                    ...学習データ、テストデータの圧縮ファイル
  ├ [Description]                       ...説明用フォルダ
  │  ├ overview.md                      ...概要ファイル
  │  └ [src]
  ├ [Leaderboard]
  │  ├ .gitkeep
  │  └ result.txt                       ...結果一覧
  ├ [Notebooks]
  │  └ [PLAYERNAME]                     ...個人フォルダ
  │    └ ...
  ├ README.md                           ...プロジェクト概要README
  └ [Submit]                            ...予測結果提出用フォルダ
    └ random_submit.csv                 ...提出データのサンプル
```

※100MB 以上の csv ファイルはコミットしないこと
プッシュしようとすると GitHub でエラーが発生します。
（Dataset、Notebooks フォルダで ignore 済み）
