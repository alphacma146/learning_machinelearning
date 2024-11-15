# %%
# standard lib
from pathlib import Path
# third party
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import plotly.express as px
# %%
train_data = pd.read_csv(Path(r"..\..\Dataset\train.csv"))
test_data = pd.read_csv(Path(r"..\..\Dataset\test.csv"))
submit_path = Path(r"..\..\Submit\sample_submit.csv")
disuse_col = ['date', 'model', 'capacity_bytes']
# %%
numeric_cols = [
    col for col in train_data.columns
    if col not in disuse_col + ["failure", "serial_number"]
]


def get_last_record(data: pd.DataFrame):
    data["date"] = pd.to_datetime(data["date"])
    ret = (
        data
        .sort_values("date")[numeric_cols + ["serial_number"]]
        .fillna(0)
        .groupby("serial_number")
        .aggregate(["last", "mean", "std", lambda x:x.max() - x.min()])
    )
    ret.columns = [
        f"{col_a}_{col_b}" if col_b != "<lambda_0>" else f"{col_a}_diff"
        for col_a, col_b in ret.columns
    ]
    return ret


train_last_data = get_last_record(train_data)
test_last_data = get_last_record(test_data)
target_data = (
    train_data
    .groupby("serial_number")
    .tail(1)
    .reset_index(drop=True)
    .set_index("serial_number")["failure"]
    .copy()
)
display(train_last_data)
display(target_data)
# %%
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components='mle', svd_solver='full')),
        ("lgrg", LogisticRegression(max_iter=1500))
    ]
)
kf = KFold(n_splits=5, shuffle=True, random_state=37)
scores = cross_val_score(
    pipe,
    train_last_data,
    target_data,
    scoring='roc_auc',
    cv=kf
)
print("roc_auc", np.average(scores))
# %%
# ROC曲線
pipe.fit(train_last_data, target_data)
fpr, tpr, _ = roc_curve(
    target_data, pipe.predict_proba(train_last_data)[:, 1]
)
fig = px.scatter(x=fpr, y=tpr)
fig.update_traces(marker_size=10)
fig.update_layout(margin={"r": 50, "t": 50, "l": 10, "b": 10}, height=450)
fig.show()
# 寄与
coef = np.abs(pipe._final_estimator.coef_.flatten())
importance_df = pd.DataFrame(
    data={
        "index": [f"pca_{i}" for i in range(1, len(coef) + 1)],
        "importance": coef
    }
).sort_values(["importance"])
fig = px.bar(importance_df, x="index", y="importance")
fig.show()
# csv出力
predict_df = pd.DataFrame(
    pipe.predict_proba(test_last_data)[:, 1],
    index=test_last_data.index,
    columns=["predict_failure"]
)
display(predict_df)
submit_df = (
    pd.merge(
        pd.read_csv(submit_path),
        predict_df,
        how="left",
        left_on="serial_number",
        right_index=True
    )
    .sort_index()
    .rename(columns={"failure": "sample_fail", "predict_failure": "failure"})
)
submit_df.to_csv(
    submit_path.parent / "submit_makabe_baseline.csv",
    index=False,
    columns=["serial_number", "failure"]
)
# %%
