# Titanic_CompeDEMO

## ☆ 参加手順 ☆

1. このレポジトリをローカルにクローン
2. 自分の名前でブランチを切る
3. `Titanic_CompeDEMO\Development`配下に自分の名前のフォルダを作る
4. 適宜コミットしながら開発する
5. 頃合いを見て`main`ブランチにマージしてプッシュ
6. 気が向いたらプルリクも投げよう

## スコアの計算方法

1. Submit フォルダに予測結果を csv 形式で保存する
2. ファイル名は分かりやすい名前に
3. `evaluation.py`を実行

    ```bash
    (venv) PS C:\Users\****\YOURWORKSPACE>python evaluation.py
    ```

    python バージョン: `Python 3.10`

    ```bash
    autopep8==2.0.2
    flake8==6.0.0
    joblib==1.2.0
    mccabe==0.7.0
    numpy==1.24.3
    pandas==2.0.1
    pycodestyle==2.10.0
    pyflakes==3.0.1
    python-dateutil==2.8.2
    pytz==2023.3
    scikit-learn==1.2.2
    scipy==1.10.1
    six==1.16.0
    threadpoolctl==3.1.0
    tomli==2.0.1
    tzdata==2023.3
    ```

4. `result.txt`に結果が吐き出される
