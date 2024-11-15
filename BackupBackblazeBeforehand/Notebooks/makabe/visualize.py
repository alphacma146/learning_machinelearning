# %%
# standard lib
from pathlib import Path
# third party
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display
pd.options.display.max_rows = 30
# %%
train_data = pd.read_csv(Path(r"..\..\Dataset\train.csv"))
test_data = pd.read_csv(Path(r"..\..\Dataset\test.csv"))
# %%


def get_last_record(data: pd.DataFrame):
    data["date"] = pd.to_datetime(data["date"])
    return (
        data
        .groupby("serial_number")
        .apply(lambda x: x.sort_values("date").tail(1))
        .reset_index(drop=True)
    )


train_last_data = get_last_record(train_data)
test_last_data = get_last_record(test_data)
model_df = (
    pd.concat(
        [
            train_last_data[["serial_number", "model"]],
            test_last_data[["serial_number", "model"]]
        ]
    )
    .groupby("model")
    .count()
    .sort_values("serial_number", ascending=False)
)
display(model_df)
# %%
series_list = [
    'TOSHIBA',
    'HGST',
    'WDC',
    'Seagate',
    'DELLBOSS',
    'CT',
    'MTF',
    'Hitachi',
    'ST'
]
df_series_dict = {
    sl: train_last_data.query(f"model.str.startswith('{sl}')")
    for sl in series_list
}


def vis_missing(df: pd.DataFrame, title: str):
    missing_df = df.isnull()
    missing_df.reset_index(inplace=True, drop=True)
    fig = px.imshow(missing_df)
    fig.update_layout(
        title={
            "text": f"{title} missing",
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
        margin={"l": 5, "b": 10},
    )
    fig.show()
    missing_sum = missing_df.sum()
    missing_sum.name = "SUM"
    missing_ratio = missing_sum / missing_df.shape[0] * 100
    missing_ratio.name = "RATIO"
    display(pd.concat([missing_sum, missing_ratio], axis=1))


for model, df in df_series_dict.items():
    vis_missing(df, f"{model}_train")
    break
miss_df = train_last_data.isna().sum()
df = pd.DataFrame(
    {"missing": miss_df, "ratio": miss_df / train_last_data.shape[0]}
).sort_values("missing", ascending=False)
miss_columns = list(df.query("ratio>=0.9").index)
print(len(miss_columns), miss_columns)
display(df)
# %%
train_last_data["series"] = train_last_data["model"].apply(
    lambda x: next((prefix for prefix in series_list if x.startswith(prefix)))
)
fail_df = (
    train_last_data[["series", "serial_number", "failure"]]
    .groupby("series")
    .aggregate({
        "serial_number": "count",
        "failure": "sum"
    })
)
fail_df["ratio"] = fail_df["failure"] / fail_df["serial_number"]
display(fail_df)
# %%
non_use_col = ["date", "serial_number", "capacity_bytes", "model", "failure"]
df = (
    train_data
    .drop(non_use_col + miss_columns, axis=1)
    .fillna(0)
)
corr_df = df.corr(method="pearson")
fig = px.imshow(corr_df)
fig.update_layout(
    margin={"l": 5, "b": 10},
    height=800
)
# 0埋めしたが、もともと0 or nanなので相関が欠損（分散が0のため）
fig.show()
std_missing_col = ["smart_23_raw", "smart_24_raw", "smart_224_raw"]
# %%


def show_scatter(data: pd.DataFrame, x_col: str, y_col: str):
    fig = px.scatter(data, x=x_col, y=y_col, color="failure")
    fig.show()


show_scatter(train_last_data, "smart_190_raw", "smart_3_raw")
# %%
df = (
    train_data
    .drop(non_use_col, axis=1)
    .fillna(0)
)
scaling = StandardScaler()
pca = PCA()
pca.fit(scaling.fit_transform(df))
ratios = pd.DataFrame({"contribution": pca.explained_variance_ratio_})
ratios["cumsum"] = ratios["contribution"].cumsum()
fig = px.line(ratios, y="cumsum")
fig.show()
# %%
