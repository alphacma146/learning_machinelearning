# BackupBackblazeBeforehand

Failure prediction of storage used for cloud backup services.

# クラウドバックアップサービスで使用されているストレージの故障予測

:arrow_forward:[まずはここをチェック](Description/overview.md)

```bash
[BackupBackblazeBeforehand]
  ├ [.git]
  ├ [.github]
  ├ .gitignore
  ├ [Dataset]                             ...データフォルダ
  │  ├ [raw_data]
  │  │  ├ backblaze_concat.py             ...データ結合&サンプリング用
  │  │  └ leftjoin_data.py                ...データ結合&サンプリング用
  │  └ resources.zip                      ...学習データ、テストデータの圧縮ファイル
  ├ [Description]                         ...説明用フォルダ
  │  ├ overview.md                        ...概要ファイル
  │  └ [src]
  ├ [Leaderboard]
  │  ├ .gitkeep
  │  └ result.txt                         ...結果一覧
  ├ [Notebooks]                           ...予測コードフォルダ
  │  └ [PLAYERNAME]                       ...個人フォルダ
  │    └ ...
  ├ README.md                             ...プロジェクト概要README
  └ [Submit]                              ...予測結果提出用フォルダ
    └ sample_submit.csv                   ...提出データのサンプル
```
