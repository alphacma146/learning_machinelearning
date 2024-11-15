# RentPredictForRental

Rent regression forecast for rental properties in Tokyo 23 special wards.

# 東京 23 区にある賃貸物件の家賃予測

:arrow_forward:[まずはここをチェック](Description/overview.md)

## フォルダ構成

```
[RentPredictForRental]
  ├ [.git]
  ├ [.github]
  ├ .gitignore                          ...gitで管理しないファイルを定義
  ├ [Dataset]                           ...データフォルダ
  │  ├ [raw_data]                       ...データ収集フォルダ
  │  │  ├ cleansing_suumo.py            ...データクレンジングコード
  │  │  ├ requirements.txt
  │  │  ├ result_suumo_finally.7z       ...スクレイピングデータの圧縮ファイル
  │  │  └ scraping_suumo.py             ...スクレイピングコード
  │  └ resources.zip                    ...学習データ、テストデータの圧縮ファイル
  ├ [Description]                       ...説明用フォルダ
  │  ├ overview.md                      ...概要ファイル
  │  └ [src]
  ├ [Leaderboard]                       ...結果一覧
  │  └ .gitkeep
  ├ [Notebooks]                         ...予測コードフォルダ
  │  └ [PLAYERNAME]                     ...個人フォルダ
  │    └ ...
  ├ README.md                           ...プロジェクト概要README
  └ [Submit]                            ...予測結果提出用フォルダ
    └ rough_submit.csv                  ...提出データのサンプル
```

※100MB 以上の csv ファイルはコミットしないこと<br>
プッシュしようとすると GitHub でエラーが発生します。<br>
（Dataset、Notebooks フォルダで ignore 済み）
