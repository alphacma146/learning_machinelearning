# S&P500 米国主要 500 社株価指数の推移予測

## ゴール

&emsp;金融データからの特徴抽出は、市場予測領域で最も重要な問題の一つであり、多くのアプローチが提案されています。自分が相場の中でどこにいるか、天井にいるのかあるは大底か、知ることは容易ではなくそのための将来的な株価指数推移予測は見知らぬ土地を歩く上での道標になります。<br>
&emsp;このデータセットには 2010 年から 2016 年までの様々なカテゴリーの技術指標、先物契約、商品価格、世界中の重要な市場指標、米国市場の主要企業の株価、そして国債利回りに関する時系列データが含まれています。参加者はこの学習データを使い、2017 年の S&P500 価格推移の予測モデルを作成します。

[Welcome to the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/554/cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables)

[CNNpred: CNN-based stock market prediction using a diverse set of variables](https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915)

## 参加手順

1. 自分の名前で`master`からブランチを切る
1. `Dataset\resources.zip`を解凍する
1. `Notebooks`配下に自分の名前でフォルダを作成、モデル開発する
1. 適宜コミット、プッシュして予測モデルをつくる
1. 予測結果は`Submit`フォルダに格納する
1. `master`マージすると結果が`Leaderboard\result.txt`に表示される
1. コードレビューしてほしいときはプルリクを投げよう

## レギュレーション

1. データ取得元の営業を妨害する行為はしないこと
2. `master`ブランチへの直接コミットは禁止
3. `master`へのマージ回数に制限なし（GitHub Actions は 2000min./月しか動けないが特に関係ない）
4. 外部データの使用は**禁止**、使えるのは`Dataset\resources.zip`からのみ
5. 使用ツールの制限はなし、なんでも OK、ただし実行手順を明確にすること（他人が実行できる手順を示すこと）
6. 提出ファイルは何枚でも OK、ファイル名は誰が提出したか判別できるようにすること

## データ

`Dataset\resources.zip`

-   `train.csv`
    学習用データ
-   `test.csv`
    テストデータ
-   `brownian_submit.csv`
    提出ファイルのサンプル
-   `Processed_S&P.csv`
    その他のデータも含む

|  column  | dtype | note   |
| :------: | :---: | :----- |
|  `Date`  | DATE  | 日付   |
| `Close`  | FLOAT | 終値   |
| `Volume` | FLOAT | 取引量 |

-   格納されているその他のデータ

|                            column                            | dtype | note                                                                                                                                                                   |
| :----------------------------------------------------------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                 `mom`,`mom1` ,`mom2` ,`mom3`                 | FLOAT | 移動平均                                                                                                                                                               |
|            `ROC_5` ,`ROC_10` ,`ROC_15` ,`ROC_20`             | FLOAT | 価格変動率                                                                                                                                                             |
|           `EMA_10` ,`EMA_20` ,`EMA_50` ,`EMA_200`            | FLOAT | 指数平滑移動平均                                                                                                                                                       |
|            `DTB4WK`, `DTB3`,`DTB6`,`DGS5`,`DGS10`            | FLOAT | 米国債利回り(DTB4WK は 4 週間国庫短期証券の利回り)                                                                                                                     |
|                        `Oil` ,`Gold`                         | FLOAT | 原油、金                                                                                                                                                               |
|                        `DAAA` ,`DBAA`                        | FLOAT | 米国 AAA 債と BAA 債の利回り                                                                                                                                           |
|                  `GBP` ,`JPY` ,`CAD` ,`CNY`                  | FLOAT | 英ポンド、日本円、カナダドル、中国人民元の為替レート                                                                                                                   |
|      `AAPL`,`AMZN`,`GE`,`JNJ`,`JPM`,`MSFT`,`WFC`,`XOM`       | FLOAT | 各企業の株価                                                                                                                                                           |
| `FCHI`,`FTSE`,`GDAXI`,`DJI`,`HSI`,`IXIC`,`SSEC`,`RUT`,`NYSE` | FLOAT | フランス CAC40、英国 FTSE100、ドイツ DAX30、米国ダウ・ジョーンズ平均株価、香港ハンセン指数、NASDAQ 総合指数、上海総合指数、ラッセル 2000、ニューヨーク証券取引所の指数 |
| `TE1`,`TE2`,`TE3`,`TE5`,`TE6`,`DE1`,`DE2`,`DE4`,`DE5`,`DE6`  | FLOAT | 米国国債（Treasury）とドイツ国債（Deutsche Bund）の利回り                                                                                                              |
|                   `CTB3M`,`CTB6M`,`CTB1Y`                    | FLOAT | 米国 3 ヶ月国債、6 ヶ月国債、1 年国債                                                                                                                                  |
|                            `Name`                            |  STR  | ターゲット名                                                                                                                                                           |
|                            その他                            | FLOAT | さまざまな通貨、指数、原料、金属、他の金融商品などの価格やレート                                                                                                       |

## 評価指標

RMSE（Root Mean Square Error）[二乗平均平方根誤差](https://ja.wikipedia.org/wiki/%E4%BA%8C%E4%B9%97%E5%B9%B3%E5%9D%87%E5%B9%B3%E6%96%B9%E6%A0%B9%E8%AA%A4%E5%B7%AE)

$RMSE=\sqrt{\dfrac{1}{n}\normalsize\displaystyle\sum_{t=1}^n(A_t-F_t)^2}$
<br>$A_t$ is the actual value and $F_t$ is the forecast value.

### 参考

sklearn<br>
[sklearn.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)

## 提出ファイルの形式

&emsp;日付ごとの終値を提出します。

```csv
Date,Close
2017-01-03,1000.00
2017-01-04,1000.00
2017-01-05,1000.00
2017-01-06,1000.00
...
```

## Deadline

## ライセンス

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

```
DOI
10.24432/C55P70
```
