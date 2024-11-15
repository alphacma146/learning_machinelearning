# 崩し字平仮名画像分類

## ゴール

&emsp;歴史的な記録文書は、過去を垣間見る機会を与えてくれます。我々は自分たちの時代の前の世界を垣間見ることができ、その文化、規範、価値観を見て、自らの考えと照らし合わせることができます。日本は非常にユニークな歴史的経路を持っています。歴史的には、日本とその文化は西洋から比較的孤立していましたが、明治維新まで、日本の指導者が教育制度を改革し、文化を近代化するための動きが起こる 1868 年まで、その状況は変わりませんでした。これにより、日本語、書記、印刷システムには大きな変化がもたらされました。この時代の日本語の近代化により、行書のくずし字は公式の学校カリキュラムから外れてしまいました。1000 年以上にわたって使用されてきたくずし字ですが、今日のほとんどの日本人は、150 年以上前に書かれたり出版されたりした本を読むことができません。<br>
&emsp;その結果、数十万のくずし字テキストがデジタル化されていますが、それらのほとんどは転写されることなく、現在ではほんの一握りの専門家しか読むことができません。私たちは、これらのテキストから手書きの文字を取り出し、それらを MNIST データセットに似た形式で前処理し、オリジナルの MNIST データセットよりも現代的で、分類が難しいベンチマークデータセットを作成しました。<br>
&emsp;これらのデータセットを公開することで、日本文学と機械学習の分野を結びつけることも期待しています 😊<br>
&emsp;📚 この論文を読むことで、くずし字、データセット、そして私たちがこれらを作成した動機についてもっと詳しく知ることができます！

[KMNIST データセット](http://codh.rois.ac.jp/kmnist/)

### 参考

[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

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
4. 外部データの使用は**OK**、使ったデータは`Dataset`フォルダに公開し取得元を明示すること
5. 使用ツールの制限はなし、なんでも OK、ただし実行手順を明確にすること（他人が実行できる手順を示すこと）
6. 提出ファイルは何枚でも OK、ファイル名は誰が提出したか判別できるようにすること

## データ

`Dataset\resources.zip`

-   `k49-train-imgs.npz`
    学習データ
-   `k49-train-labels.npz`
    学習データラベル
-   `k49_classmap.csv`
    ラベルとひらがなの変換テーブル
-   `k49-test-imgs.npz`
    テストデータ
-   `random_submit.csv`
    提出ファイルサンプル、乱数で予測したデータ

### npz ファイルの読み込み方法

```python
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

image_data = np.load(Path(r"..\..\Dataset\k49-train-imgs.npz"))
print(image_data)
# >> NpzFile '..\\..\\Dataset\\k49-train-imgs.npz' with keys: arr_0
img_data_arr = image_data["arr_0"]
sns.heatmap(img_data_arr[0], cmap='cividis', cbar=False)
plt.axis('off')
plt.show()
```

## 評価指標

マクロ平均 F1 スコア
[F1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

$${F1}=\frac{2\times\text{True Positive}}{2\times\text{True Positive}+\text{False Positive}+\text{FalseNegative}}$$

## 提出ファイルの形式

分類した平仮名のラベルを提出する。

```csv
index,label
0,0
1,0
2,0
3,0
4,0
5,0
6,0
7,0
8,0
9,0
```

## Deadline

## ライセンス

```
『KMNISTデータセット』（CODH作成） 『日本古典籍くずし字データセット』（国文研ほか所蔵）を翻案
doi:10.20676/00000341
```
