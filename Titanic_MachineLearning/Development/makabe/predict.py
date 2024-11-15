# %%
# Standard lib
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import optuna.integration.lightgbm as opt_lgb
import plotly.express as px
from IPython.display import display


@dataclass
class ConstData():
    model_params: dict = field(default_factory=lambda: {
        "device": "gpu",
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "binary_logloss",
        "max_bins": 255,
        "learning_rate": 0.01
    })
    sex: dict = field(default_factory=lambda: {
        "male": 0,
        "female": 1
    })
    embarked: dict = field(default_factory=lambda: {
        "C": 1,
        "Q": 2,
        "S": 3
    })
    cabin: dict = field(default_factory=lambda: {
        "m": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "T": 8
    })
    honor: dict = field(default_factory=lambda: {
        'Capt': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Don': 'Royalty',
        'Sir': 'Royalty',
        'the Countess': 'Royalty',
        'Dona': 'Royalty',
        'Lady': 'Royalty',
        'Mme': 'Mrs',
        'Ms': 'Mrs',
        'Mrs': 'Mrs',
        'Mlle': 'Miss',
        'Miss': 'Miss',
        'Mr': 'Mr',
        'Master': 'Master',
        'Jonkheer': 'Master'
    })
    honor_category: dict = field(default_factory=lambda: {
        'Officer': 1,
        'Royalty': 2,
        'Mrs': 3,
        'Miss': 4,
        'Mr': 5,
        'Master': 6
    })


CFG = ConstData()

# %%
# data load
train_data = pd.read_csv(
    Path(r"..\..\Document\data\train.csv"), index_col="PassengerId"
)
test_data = pd.read_csv(
    Path(r"..\..\Document\data\test.csv"), index_col="PassengerId"
)
categorical_columns = ["Pclass", "Sex", "Embarked", "Floor"]
# %%
# labeling and fill NA


def split_honor(name: str):
    """名前から称号を取り出す"""
    ret = name.split(",")[-1].strip(" ")
    ret = ret.split(". ")[0]
    return ret


for df in (train_data, test_data):
    df.replace(CFG.sex, inplace=True)
    df.replace(CFG.embarked, inplace=True)
    df["miss_Age"] = df["Age"].isna().astype(int)
    df["miss_Cabin"] = df["Cabin"].isna().astype(int)
    df["Family"] = df["SibSp"] + df["Parch"]
    df["Fare"] = np.log(df[["Fare"]] + 1)
    df["Surname"] = df["Name"].apply(lambda x: x.split(',')[0].strip())
    df["Honor"] = df["Name"].apply(split_honor).replace(CFG.honor)
# %%
# duplicate ticket
all_data = pd.concat([train_data, test_data])
ticket_dict = all_data["Ticket"].value_counts().to_dict()
all_data["Leaves"] = (
    all_data["Ticket"]
    .apply(lambda x: ticket_dict.get(x))
    .astype(int)
)
# ticket alpha num


def sep_ticket(ticket: str):
    it = ticket.split(" ")
    if len(it) == 1:
        alpha = 0
    else:
        alpha = 1
    return alpha, len(it[-1])


all_data[["isalp_Ticket", "length_Ticket"]] = all_data.apply(
    lambda x: sep_ticket(x["Ticket"]), axis=1, result_type="expand"
)
# surname
surname_dict = train_data["Surname"].value_counts().to_dict()
survived_dict = (
    train_data[["Survived", "Surname"]]
    .groupby("Surname")
    .mean()
    .to_dict()
)["Survived"]
sur_list = []
die_list = []
for k, i in surname_dict.items():
    if i == 1:
        continue
    rat = survived_dict.get(k)
    if rat == 1.0:
        sur_list.append(k)
    elif rat == 0.0:
        die_list.append(k)
    else:
        pass
all_data["sur_Relative"] = all_data["Surname"].apply(
    lambda x: 1 if x in sur_list else 0
)
all_data["die_Relative"] = all_data["Surname"].apply(
    lambda x: 1 if x in die_list else 0
)
# cabin


def init_cabin(cabin: str):
    """客室のイニシャル、複数部屋ならより深い方"""
    if pd.isnull(cabin):
        ret = cabin
    else:
        ret = sorted([alpha for alpha in cabin.split(" ")])[-1]
        ret = ret[0]
    return ret


all_data["Floor"] = (
    all_data["Cabin"]
    .apply(init_cabin)
    .replace(CFG.cabin)
)
res_list = []
imputer = KNNImputer(n_neighbors=3)
use_col = ["Fare", "Embarked", "Floor"]
for pclass in all_data["Pclass"].unique():
    df = all_data.query(f"Pclass=={pclass}")[use_col]
    res = pd.DataFrame(
        data=imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    res_list.append(res)
res_df = pd.concat(res_list)
all_data[use_col] = res_df[use_col]
all_data[["Embarked", "Floor"]] = all_data[["Embarked", "Floor"]].round()
for col in categorical_columns:
    all_data[col] = all_data[col].astype(int)
# Age
res_list = []
use_col = ["Age", "Pclass", "Fare", "Embarked", "Floor"]
for honor in all_data["Honor"].unique():
    df = all_data.query(f"Honor=='{honor}'")[use_col]
    train_X = df.query("Age==Age")
    target_X = df.query("Age!=Age")
    if len(target_X) == 0:
        continue
    estimator = RandomForestRegressor(
        random_state=37, n_estimators=100, n_jobs=-1
    )
    estimator.fit(train_X.drop("Age", axis=1), train_X["Age"])
    res = pd.DataFrame(
        data={
            "Age": estimator.predict(target_X.drop("Age", axis=1))
        },
        index=target_X.index,
    )
    res_list.append(res)
res_df = pd.concat(res_list).sort_index()
all_data.loc[res_df.index, "Age"] = res_df["Age"].round()
# %%
# preprocess
col = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "miss_Age",
    "miss_Cabin",
    "Family",
    "Honor",
    "Leaves",
    "Floor",
    "sur_Relative",
    "die_Relative"
]
cat_col = [
    "Pclass",
    "Embarked",
    "cat_Age",
    "cat_Honor",
    "cat_Family",
    "cat_Leaves",
    "cat_Floor"
]
train_df = all_data.loc[train_data.index][col]
test_df = all_data.loc[test_data.index][col]
scaler = StandardScaler()
scaler.fit(train_df[["Age", "Fare"]])
train_labels = train_df["Survived"].astype(int).copy()
for df in (train_df, test_df):
    df.drop("Survived", axis=1, inplace=True)
    df["Alone"] = df["Family"].apply(lambda x: 1 if x == 0 else 0)
    df["cat_Age"] = df["Age"].apply(lambda x: x // 20).astype(int)
    df["cat_Age"] .where(df["cat_Age"] <= 2, 3, inplace=True)
    df["cat_Honor"] = df["Honor"].apply(
        lambda x: 0 if x in ("Mr", "Officer") else 1 if x in ("Master") else 2
    )
    df["cat_Family"] = df["Family"].apply(
        lambda x: 0 if x >= 7 else 2 if 1 <= x <= 3 else 1
    )
    df["cat_Leaves"] = df["Leaves"].apply(
        lambda x: 0 if 5 <= x <= 6 else 2 if 2 <= x <= 4 else 1
    )
    df["cat_Floor"] = df["Floor"].apply(
        lambda x: 0 if 6 <= x <= 8 else 2 if x == 2 or x == 5 else 1
    )
    # df["is_Fare"] = (df["Fare"] >= 3.9).astype(int)
    df.replace(CFG.honor_category, inplace=True)
    df.drop(
        ["Honor", "Family", "Leaves", "Floor"],
        axis=1,
        inplace=True
    )
    # df.loc[:, ("Age", "Fare")] = pd.DataFrame(
    #    scaler.transform(df[["Age", "Fare"]]),
    #    index=df.index,
    #    columns=["Age", "Fare"]
    # )
    df.drop(
        ["Age", "Fare"],
        axis=1,
        inplace=True
    )
train_df = pd.get_dummies(train_df, columns=cat_col)
test_df = pd.get_dummies(test_df, columns=cat_col)
# %%
# lgb parameter tuning
""" tuner = opt_lgb.LightGBMTunerCV(
    CFG.model_params,
    train_set=lgb.Dataset(train_df, train_labels),
    num_boost_round=500,
    # categorical_feature=cat_col,
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(0),
    ]
)
tuner.run()
print(tuner.best_params)
print(tuner.best_score) """
# lightgbm model
""" params = CFG.model_params | tuner.best_params
lgb_model = lgb.LGBMClassifier(**params, num_iterations=1000)
lgb_model.fit(
    train_df,
    train_labels,
    # categorical_feature=cat_col,
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(10),
    ]
)
importance = pd.DataFrame(
    data={
        "col": train_df.columns,
        "importance": lgb_model.feature_importances_
    }
).sort_values(["importance"])
fig = px.bar(importance, x="col", y="importance")
fig.show() """
# %%
# RandomForest
pipeline = make_pipeline(
    RandomForestClassifier(random_state=10, max_features='sqrt')
)
param_test = {
    "randomforestclassifier__n_estimators": list(range(20, 30, 1)),
    "randomforestclassifier__max_depth": list(range(3, 10, 1))
}
gsearch = GridSearchCV(
    estimator=pipeline,
    param_grid=param_test,
    scoring='accuracy',
    cv=10
)
gsearch.fit(train_df, train_labels)
print(gsearch.best_params_, gsearch.best_score_)
# %%
estimator = RandomForestClassifier(
    random_state=10,
    warm_start=True,
    n_estimators=gsearch.best_params_["randomforestclassifier__n_estimators"],
    max_depth=gsearch.best_params_["randomforestclassifier__max_depth"],
    max_features='sqrt'
)
selector = RFECV(estimator, min_features_to_select=25, step=1, cv=10)
selector.fit(train_df, train_labels)
# cv_score = cross_val_score(selector, train_df, train_labels, cv=10)
res = selector.cv_results_
ave_score = round(np.mean(res["mean_test_score"]), 4)
std_score = round(np.mean(res["std_test_score"]), 4)
print("CV Score".ljust(10, " ") + f"Ave - {ave_score}")
print("".ljust(10, " ") + f"Std - {std_score}")
display(pd.DataFrame({
    "feature": selector.feature_names_in_,
    "support": selector.support_,
    "ranking": selector.ranking_
}))
# %%
# output
result = pd.DataFrame(
    data={
        # "lgb": lgb_model.predict_proba(test_df)[:, 1],
        "rf": selector.predict_proba(test_df)[:, 1]
    },
    index=test_df.index
)
# result["Survived"] = result.mean(axis=1).round().astype(int)
result["Survived"] = (result["rf"] >= 0.5).astype(int)
display(result)
result[["Survived"]].to_csv(Path("result_submission.csv"))
# %%
