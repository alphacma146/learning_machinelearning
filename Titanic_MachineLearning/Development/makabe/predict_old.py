# %%
# Standard lib
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import metrics, model_selection

sns.set(style="darkgrid")
# %%


@dataclass
class ConstData():
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
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "T": 7
    })
    honor: dict = field(default_factory=lambda: {
        'Mr.': 1,
        'Mrs.': 0,
        'Miss.': 0,
        'Master.': 0,
        'Oth.': 1
    })


CD = ConstData()

# %%
# data load
train_data = pd.read_csv(
    Path(r"..\..\Document\data\train.csv"), index_col="PassengerId"
)
test_data = pd.read_csv(
    Path(r"..\..\Document\data\test.csv"), index_col="PassengerId"
)
# %%
# labeling and fill NA
for data in (train_data, test_data):
    data.replace(CD.sex, inplace=True)
    data.replace(CD.embarked, inplace=True)
    data.replace(CD.honor, inplace=True)
    for index, items in data.query("Cabin==Cabin").iterrows():
        it = items["Cabin"]
        data.at[index, "Floor"] = np.mean([
            CD.cabin[i[0]] for i in (
                it.split(" ") if " " in it else [it]
            )
        ])
    data.drop(
        ["Ticket", "Name", "Cabin", "Fare"],
        axis=1,
        inplace=True
    )
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
print(train_data.head())
sns.catplot(
    x="Floor",
    y="Survived",
    hue="Pclass",
    data=train_data,
    kind="bar",
    aspect=2
)
plt.show()
# %%


def valid_model(
    model,
    args: list,
    method: str = None
):
    """_summary_

    Args:
        model (_type_): _description_
        args (list): _description_
        method (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    result_list = []
    for _ in range(100):
        data = model_selection.train_test_split(*args, test_size=0.1)
        train_x, test_x, train_y, test_y = data
        model.fit(train_x, train_y)
        match method:
            case "accuracy":
                score = metrics.accuracy_score(
                    test_y, model.predict(test_x)
                )
            case "mean_squared":
                score = metrics.mean_squared_error(
                    test_y, model.predict(test_x)
                )
        result_list.append((model, score))

    match method:
        case "accuracy":
            ret = max(result_list, key=lambda x: x[1])
        case "mean_squared":
            ret = min(result_list, key=lambda x: x[1])

    return ret
# %%


def fill_na(
    elem: str,
    train: pd.DataFrame,
    replace_list: tuple[pd.DataFrame],
    remove: str = None,
    objective=None
):
    """_summary_

    Args:
        elem (str): _description_
        train (pd.DataFrame): _description_
        replace_list (tuple[pd.DataFrame]): _description_
        remove (str, optional): _description_. Defaults to None.
        objective (_type_, optional): _description_. Defaults to None.
    """
    args = (train.drop(elem, axis=1), train[elem])
    match objective:
        case "regressor":
            model_lgb, score = valid_model(
                lgb.LGBMRegressor(),
                args,
                "mean_squared"
            )
        case "classifier":
            model_lgb, score = valid_model(
                lgb.LGBMClassifier(),
                args,
                "accuracy"
            )
    print(
        f"{elem}".ljust(7, "_")
        + f": {score:.3f}"
        + "[-]"
    )
    lgb.plot_importance(model_lgb)
    plt.show()
    for data in replace_list:
        if "Survived" in data.columns:
            temp_data = data.drop("Survived", axis=1)
        else:
            temp_data = data
        target_data = temp_data.query(f"{elem}!={elem}").drop(
            elem if remove is None else [elem, remove],
            axis=1
        )
        for index, pred in zip(
            target_data.index,
            model_lgb.predict(target_data)
        ):
            data.at[index, elem] = pred


# %%
# Child and floor
child_train = pd.concat([
    train_data.drop("Survived", axis=1).query("Age==Age"),
    test_data.query("Age==Age")
]).reindex(columns=train_data.columns[1:])
child_train["Child"] = child_train["Age"].apply(lambda x: 1 if x <= 15 else 0)
train_data.insert(3, "Child", np.nan)
test_data.insert(3, "Child", np.nan)
fill_na(
    "Child",
    child_train.drop("Age", axis=1),
    (train_data, test_data),
    "Age",
    "classifier"
)
cabin_train = pd.concat([
    train_data.drop("Survived", axis=1).query("Floor==Floor"),
    test_data.query("Floor==Floor")
]).reindex(columns=train_data.columns[1:])
fill_na(
    "Floor",
    cabin_train.drop("Age", axis=1),
    (train_data, test_data),
    "Age",
    "regressor"
)
# %%
# one-hot encoding
res_data = []
for data in (train_data, test_data):
    data["Alone"] = data.apply(
        lambda x: 0 if x["SibSp"] + x["Parch"] == 0 else 1,
        axis=1
    )
    data = data.astype({"Pclass": int, "Embarked": int, "Child": int})
    data = pd.get_dummies(data, columns=["Pclass", "Embarked"])
    res_data.append(data)
train_data, test_data = res_data
# %%
# learning
remove = ["Age", "SibSp", "Parch", "Floor"]
grid_search = []
# num_lev=13, min_leaf=7
""" for num_lev, min_leaf in tqdm([
    (num_lev, min_leaf)
    for num_lev in range(3, 20)
    for min_leaf in range(1, 10)
]): """
params = {
    "task": "train",
    "boosting": "gbdt",
    "objective": "binary",
    "metric": {"binary_logloss"},
    'num_leaves': 13,
    'min_data_in_leaf': 7,
    'max_depth': -1,
    "force_col_wise": True,
    "device": "gpu"
}
for _ in range(100):
    train, valid = model_selection.train_test_split(
        train_data.drop(remove, axis=1), test_size=0.1
    )
    train_lgb = lgb.Dataset(
        train.drop("Survived", axis=1),
        train["Survived"]
    )
    valid_lgb = lgb.Dataset(
        valid.drop("Survived", axis=1),
        valid["Survived"]
    )
    model_lgb = lgb.train(
        params,
        train_lgb,
        valid_sets=valid_lgb,
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0),
        ]
    )
    result = model_lgb.predict(valid.drop("Survived", axis=1))
    score = metrics.accuracy_score(
        valid["Survived"],
        np.where(result < 0.5, 0, 1)
    )
    grid_search.append((model_lgb, params, result, score))
    print(score)
# %%
# output
best_model, best_params, best_result, best_score = max(
    grid_search, key=lambda x: x[3]
)
sns.displot(best_result, bins=20, kde=True, rug=False)
plt.show()
lgb.plot_importance(best_model)
plt.show()
print(best_score, best_params, sep="\n")
result = pd.DataFrame(
    data={
        "Survived": np.where(
            best_model.predict(test_data.drop(remove, axis=1)) < 0.5,
            0, 1
        )
    },
    index=test_data.index
)
result.to_csv(Path("result_submission.csv"))
len(pd.merge(
    pd.read_csv(
        Path("gender_submission.csv"),
        index_col="PassengerId"
    ),
    result,
    on="PassengerId",
    suffixes=(col := ["_gender", "_result"])
).query(f"Survived{col[0]}!=Survived{col[1]}").index)
# %%
