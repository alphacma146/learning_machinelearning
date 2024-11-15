# %%
# Standard lib
from pathlib import Path
from collections import Counter
# Third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.impute import KNNImputer
from IPython.display import display
sns.set(style="darkgrid")
# %%
# %% [markdown]
# ### Train Data Column
# |Column|Note|
# |:--:|:--:|
# |PassengerId|unique key|
# |<font color="red">Survived</font>|0 or 1|
# |Pclass|1:higher,2:middle,3:lower|
# |Name|name|
# |Sex|gender|
# |Age|age|
# |SibSp|brother or spouse|
# |Parch|parents or children|
# |Ticket|ticket ID|
# |Fare|fare|
# |Cabin|room number|
# |Embarked|C:Cherboug,Q:Queenstown,S:Southamptom|
# %%
train_data = pd.read_csv(
    Path(r"..\..\Document\data\train.csv"), index_col="PassengerId"
)
test_data = pd.read_csv(
    Path(r"..\..\Document\data\test.csv"), index_col="PassengerId"
)

display(train_data.head(10))
display(train_data.describe())
display(test_data.head(10))
display(test_data.describe())
# %%
sns.jointplot(
    x="Age",
    y="Fare",
    data=train_data,
    kind="hex"
)
plt.show()
sns.catplot(
    x="Sex",
    y="Age",
    hue="Survived",
    data=train_data,
    kind="swarm",
    aspect=1.2
)
plt.show()
sns.catplot(
    x="Embarked",
    y="Survived",
    hue="Pclass",
    data=train_data,
    kind="bar"
)
plt.show()

# Pclass
Embarked_unique = train_data.query("Embarked==Embarked")["Embarked"].unique()
Embarked_df_list = [
    pd.DataFrame(data={
        f"{i}": (
            train_data
            .query(f"Embarked=='{i}'")["Pclass"]
            .value_counts()
            .sort_index()
        )
    }).T for i in Embarked_unique
]
Embarked_df = pd.concat(Embarked_df_list)
table_df = Embarked_df.join(
    pd.DataFrame(
        {"Total": (
            train_data
            .query("Embarked==Embarked")["Embarked"]
            .value_counts()
        )}
    )
)
display(table_df)
# %%
# train missing


def vis_missing(df: pd.DataFrame, title: str):
    missing_df = df.isnull()
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


for args in (
    (train_data, "train"),
    (test_data, "test")
):
    vis_missing(*args)

display(test_data.query("Fare!=Fare"))
display(train_data.query("Embarked!=Embarked"))
# %%
# replace


def init_cabin(cabin: str):
    ret = sorted([alpha for alpha in cabin.split(" ")])[-1]
    return ret[0]


sex: dict = {
    "male": 0,
    "female": 1
}
embarked: dict = {
    "C": 1,
    "Q": 2,
    "S": 3
}
cabin: dict = {
    "m": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "T": 8
}
replace_data = (
    train_data
    .replace(sex)
    .replace(embarked)
)
replace_data["miss_Age"] = train_data["Age"].isna().astype(int)
replace_data["miss_Cabin"] = train_data["Cabin"].isna().astype(int)
replace_data["Floor"] = (
    replace_data["Cabin"]
    .fillna("m")
    .apply(init_cabin)
    .replace(cabin)
)
replace_data["Family"] = train_data["SibSp"] + train_data["Parch"]
replace_data["Alone"] = replace_data["Family"].apply(
    lambda x: 1 if x == 0 else 0
)
ticket_dict = (
    replace_data["Ticket"]
    .value_counts()
    .to_dict()
)
replace_data["Leaves"] = (
    replace_data["Ticket"]
    .apply(lambda x: ticket_dict.get(x))
    .astype(int)
)
# correlation
corr_matrix = (
    replace_data
    .fillna(0)
    .drop(["Name", "Ticket", "Cabin"], axis=1)
    .corr("spearman")
)
corr_matrix.where(corr_matrix.abs() >= 0.1, 0, inplace=True)
fig = px.imshow(corr_matrix)
fig.update_layout(
    title={
        "text": "correlation matrix",
        "font": {"size": 22, "color": "black"},
        "x": 0.5,
        "y": 0.95,
    },
    margin={"l": 5, "b": 10},
    width=700,
    height=600,
)
fig.show()
# %%


def split_honor(name: str):
    ret = name.split(",")[-1].strip(" ")
    ret = ret.split(". ")[0]
    return ret


def mortality(data: pd.DataFrame, target_col: str):
    ratio = data[["Survived", target_col]].groupby(target_col).sum()
    count = data[target_col].value_counts()
    df = (
        pd
        .concat([ratio, count], axis=1)
        .sort_values("count", ascending=False)
    )
    df["Ratio"] = (1 - df["Survived"] / df["count"]) * 100
    display(df)
    fig = px.bar(df, x=df.index, y="Ratio")
    fig.update_layout(
        title={
            "text": f"Mortality Rate by {target_col}",
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
        margin={"l": 5, "b": 10},
    )
    fig.show()


honor_dict = {}
honor_dict.update(dict.fromkeys(
    ['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
honor_dict.update(dict.fromkeys(
    ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
honor_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
honor_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
honor_dict.update(dict.fromkeys(['Mr'], 'Mr'))
honor_dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
print(honor_dict)
train_data["Honor"] = train_data["Name"].apply(split_honor).replace(honor_dict)
fig = px.histogram(train_data, x="Honor", color="Survived")
fig.show()
mortality(train_data, "Honor")
ave_age = train_data[["Age", "Honor"]].groupby("Honor").mean()
display(ave_age)
fig = px.box(train_data, x="Honor", y="Age", points="all")
fig.show()
# %%
fare_data = np.log(train_data[["Fare"]] + 1)
fig = px.histogram(fare_data)
fig.update_layout(
    title={
        "text": "Fare",
        "font": {"size": 22, "color": "black"},
        "x": 0.5,
        "y": 0.95,
    },
    margin={"l": 5, "b": 10},
)
fig.show()

# %%
fig = px.histogram(replace_data, x="Family", color="Survived")
fig.show()
mortality(replace_data, "Family")
# %%
fig = px.histogram(replace_data, x="Leaves", color="Survived")
fig.show()
mortality(replace_data, "Leaves")
# %%
df = pd.concat([train_data, test_data])
split_ticket = [
    i.replace(".", "") for it in
    [it.split(" ") for it in df["Ticket"].to_list()]
    for i in it
    if not i.isdigit()
]
display(Counter(split_ticket))
wordcloud = WordCloud(
    background_color="white",
    width=800,
    height=600,
    stopwords=[],
    font_path="C:\\Windows\\Fonts\\yumin.ttf"
)
wordcloud.generate(" ".join(split_ticket))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

train_data["feat_Ticket"] = train_data["Ticket"].apply(
    lambda x: 1 if len(x.split(" ")) != 1 else 0
)
fig = px.histogram(train_data, x="feat_Ticket", color="Survived")
fig.show()
# %%
# knn欠損埋め


def knn_naimputer(data: pd.DataFrame):
    """
    fill in missing value
    """
    imputer = KNNImputer(n_neighbors=3)
    return imputer.fit_transform(data)


use_df = replace_data[["Pclass", "Fare", "Floor", "Embarked"]]
use_df["Floor"].where(use_df["Floor"] != 0, np.nan, inplace=True)
res_df = pd.DataFrame(
    data=knn_naimputer(use_df),
    index=use_df.index,
    columns=use_df.columns
)
res_df["Floor"] = res_df["Floor"].round()
fig = px.histogram(replace_data, x="Floor", color="Pclass")
fig.show()
fig = px.histogram(res_df, x="Floor", color="Pclass")
fig.show()
# %%
# Pclassごとに欠損埋め
res_list = []
for pclass in (1, 2, 3):
    df = use_df.query(f"Pclass=={pclass}")
    res = pd.DataFrame(
        data=knn_naimputer(df),
        index=df.index,
        columns=df.columns
    )
    res_list.append(res)
res_df = pd.concat(res_list)
res_df["Floor"] = res_df["Floor"].round()
display(res_df)
fig = px.histogram(replace_data, x="Floor", color="Pclass")
fig.show()
fig = px.histogram(res_df, x="Floor", color="Pclass")
fig.show()
replace_data["Floor"] = res_df["Floor"]
mortality(replace_data, "Floor")
# %%


def sep_ticket(ticket: str):
    it = ticket.split(" ")
    if len(it) == 1:
        alpha = 0
    else:
        alpha = 1
    return alpha, len(it[-1])


df = train_data[["Survived", "Age", "Ticket", "Fare"]].copy()
df[["alpha", "length"]] = df.apply(
    lambda x: sep_ticket(x["Ticket"]), axis=1, result_type='expand'
)
df["log_Fare"] = np.log(df["Fare"] + 1)
df["log_Age"] = np.log(df["Age"] + 1)
df["times"] = (
    (df["log_Fare"] >= 4).astype(int)
    + (df["log_Age"] <= 2).astype(int)
)
display(df)
fig = px.scatter(df, x="length", y="Fare", color="Survived")
fig.show()
fig = px.histogram(df, x="length", color="Survived")
fig.show()
mortality(df, "length")
fig = px.scatter(df, x="log_Fare", y="log_Age", color="Survived")
fig.show()
mortality(df, "times")
# %%
