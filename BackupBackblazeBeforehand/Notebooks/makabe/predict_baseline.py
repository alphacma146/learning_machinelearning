# %%
# standard lib
from pathlib import Path
# third party
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
import plotly.express as px
# %%
PARAM_SEARCH = True
train_data = pd.read_csv(Path(r"..\..\Dataset\train.csv"))
test_data = pd.read_csv(Path(r"..\..\Dataset\test.csv"))
submit_path = Path(r"..\..\Submit\sample_submit.csv")
disuse_col = ['date', 'model', 'capacity_bytes']
miss_columns = [  # 欠損率9割以上
    'smart_11_raw',
    'smart_13_raw',
    'smart_15_raw',
    'smart_16_raw',
    'smart_17_raw',
    'smart_168_raw',
    'smart_170_raw',
    'smart_173_raw',
    'smart_174_raw',
    'smart_177_raw',
    'smart_179_raw',
    'smart_181_raw',
    'smart_182_raw',
    'smart_183_raw',
    'smart_201_raw',
    'smart_218_raw',
    'smart_225_raw',
    'smart_231_raw',
    'smart_232_raw',
    'smart_233_raw',
    'smart_235_raw',
    'smart_250_raw',
    'smart_251_raw',
    'smart_252_raw',
    'smart_254_raw',
    'smart_255_raw'
] + ["smart_23_raw", "smart_24_raw", "smart_224_raw"]  # 分散0
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
# %%
numeric_cols = [
    col for col in train_data.columns
    if col not in disuse_col + miss_columns + ["failure", "serial_number"]
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

""" def get_last_record(data: pd.DataFrame):
    data["date"] = pd.to_datetime(data["date"])
    return (
        data
        .groupby("serial_number")
        .apply(lambda x: x.sort_values("date").tail(1))
        .reset_index(drop=True)
        .set_index("serial_number")
        .drop(disuse_col + miss_columns, axis=1)
        .fillna(0)
    )


train_data["series"] = train_data["model"].apply(
    lambda x: next((prefix for prefix in series_list if x.startswith(prefix)))
)
train_last_data = get_last_record(train_data)
test_last_data = get_last_record(test_data)
target_data = train_last_data["failure"].copy()
stratified_data = train_last_data["series"].copy()
train_last_data.drop(["failure", "series"], axis=1, inplace=True)
display(train_last_data) """
# %%
params = {
    "device": "gpu",
    "objective": "binary",
    "boosting_type": "gbdt",
    "max_bins": 255,
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "verbose": -1,
}
kf = KFold(n_splits=5, shuffle=True, random_state=37)
# tuning
match PARAM_SEARCH:
    case True:
        tuner = opt_lgb.LightGBMTunerCV(
            params,
            train_set=lgb.Dataset(train_last_data, target_data),
            num_boost_round=100,
            shuffle=True,
            folds=kf,
            callbacks=[
                lgb.early_stopping(10),
                lgb.log_evaluation(0),
            ]
        )
        tuner.run()
        params = tuner.best_params
    case False:
        params = {
            'device': 'gpu',
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'max_bins': 255,
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'verbose': -1,
            'feature_pre_filter': False,
            'lambda_l1': 1.2726889192390749e-06,
            'lambda_l2': 5.219812473416002e-06,
            'num_leaves': 85,
            'feature_fraction': 0.5,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 20
        }
        # roc_auc_score 0.9004538933958453
# %%
score_list = []
train_predict = []
test_predict = []
importance = []
kf = KFold(n_splits=5, shuffle=True, random_state=37)
for fold, (train_idx, validate_idx) in enumerate(
    kf.split(train_last_data),
    start=1
):
    lgb_model = lgb.LGBMClassifier(
        **params,
        n_estimators=2000,
    )
    lgb_model.fit(
        train_last_data.iloc[train_idx],
        target_data.iloc[train_idx],
        eval_set=(
            train_last_data.iloc[validate_idx],
            target_data.iloc[validate_idx],
        ),
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0),
        ]
    )
    score = roc_auc_score(
        target_data.iloc[validate_idx],
        lgb_model.predict_proba(train_last_data.iloc[validate_idx])[:, 1]
    )
    score_list.append(score)
    test_predict.append(lgb_model.predict_proba(test_last_data)[:, 1])
    train_predict.append(lgb_model.predict_proba(train_last_data)[:, 1])
    importance.append(lgb_model.feature_importances_)
    print(fold, score)
# %%
# ROC曲線
print(roc_auc_score.__name__, np.average(score_list))
fpr, tpr, _ = roc_curve(
    target_data,
    np.average(train_predict, axis=0)
)
fig = px.scatter(x=fpr, y=tpr)
fig.update_traces(marker_size=10)
fig.update_layout(margin={"r": 50, "t": 50, "l": 10, "b": 10}, height=450)
fig.show()
# 寄与
importance_df = pd.DataFrame(
    data={
        "col": test_last_data.columns,
        "importance": np.average(importance, axis=0)
    }
).sort_values(["importance"])
fig = px.bar(importance_df, x="col", y="importance")
fig.show()
# csv出力
predict_df = pd.DataFrame(
    np.average(test_predict, axis=0),
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
display(importance_df)
print(params)
# %%
