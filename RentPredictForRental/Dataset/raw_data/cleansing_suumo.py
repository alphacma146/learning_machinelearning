# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from IPython.display import display


# %%
file_list = []
for file_path in Path(__file__).parent.glob("*.csv"):
    target = Path(file_path)
    ward_df = pd.read_csv(target, index_col=0)
    ward_df["ward"] = target.stem.split("_")[-1]
    file_list.append(ward_df)
df = pd.concat(file_list)
display(df.info())
# 重複削除
duplicated_col = [
    'rent',
    'fee',
    '間取り',
    '専有面積',
    '向き',
    '建物種別',
    '築年数',
    'location',
    'option',
    '間取り詳細',
    '構造',
    '階建',
    '築年月',
    '駐車場',
    '条件',
    '総戸数',
]
df.sort_values(duplicated_col, inplace=True)
dup_df = df.duplicated(subset=duplicated_col, keep=False)
print(dup_df.sum())
df.drop_duplicates(subset=duplicated_col, keep="last", inplace=True)
record_size = df.shape[0]
# %%
# index振り直し
df_drop = (
    df
    .sample(frac=1)
    .reset_index(drop=True)
    .reset_index()
)
# 数値化
df_drop["rent"] = df_drop["rent"].apply(
    lambda x: int(float(x.rstrip("万円")) * 10_000)
)
df_drop["fee"] = df_drop["fee"].apply(
    lambda x: float(x.rstrip("円")) if x == x else x
)
df_drop["専有面積"] = df_drop["専有面積"].apply(
    lambda x: float(x.replace("m2", "")) if x == x else x
)
df_drop = df_drop.rename(
    columns={
        "間取り": "room_layout",
        "専有面積": "floor_area",
        "向き": "facing",
        "建物種別": "type",
        "築年数": "age_of_construction",
        "間取り詳細": "room_layout_detail",
        "構造": "structure",
        "階建": "floor",
        "築年月": "year_of_construction",
        "駐車場": "parking",
        "条件": "conditions",
        "総戸数": "total_rooms",
        "周辺情報": "neighborhood_information",
    }
)
display(df_drop.isnull().sum(), df_drop.dtypes)
display(df_drop.head(3))
# %%
val_cnt = (
    df_drop[["rent", "ward"]]
    .groupby("ward")
    .aggregate(["count", "std"])
)
val_cnt["layer"] = val_cnt[('rent', 'count')] * val_cnt[('rent', 'std')]
val_cnt["allocate"] = val_cnt["layer"] / val_cnt["layer"].sum()
val_cnt["sampling"] = (
    val_cnt["allocate"] * (record_size * 0.05)
).round().astype("int64")
val_cnt["rate"] = val_cnt["sampling"] / val_cnt[('rent', 'count')]
display(val_cnt)
# 分割
df_count = df_drop[["location"]].value_counts()
df_count.columns = ["count"]
df_merge = df_drop.merge(df_count, on="location").set_index("index", drop=True)
severe_duplicated_col = [
    'rent',
    'fee',
    'room_layout',
    'floor_area',
    'facing',
    'type',
    'age_of_construction',
    'location',
    'structure',
    'floor',
    'year_of_construction',
    'total_rooms',
]
dup_df = df_merge.duplicated(subset=severe_duplicated_col, keep=False)
print(dup_df.sum())
df_unique = df_merge.drop_duplicates(
    subset=severe_duplicated_col,
    keep=False
)
target_df = df_unique.query("count!=1")
location_df = df_unique.query("count==1")
# %%
test_list = []
sampling_dict = val_cnt["sampling"].to_dict()
for ward in target_df["ward"].unique():
    num = sampling_dict[ward]
    df = target_df.query(f'ward=="{ward}"')
    _, test = train_test_split(df, test_size=num, random_state=3)
    test_list.append(test)
_, test = train_test_split(location_df, test_size=15, random_state=3)
test_list.append(test)
test_df = pd.concat(test_list)
train_df = df_merge.loc[~df_merge.index.isin(test_df.index)]
print(train_df.shape)
print(test_df.shape)
# (575254, 21)
# (30292, 21)
# %%
train = (
    train_df.drop(["fee", "ward", "count"], axis=1)
    .sort_index()
)
train.to_csv(Path("../train.csv"))
test = (
    test_df.drop(["rent", "fee", "ward", "count"], axis=1)
    .sort_index()
)
test.to_csv(Path("../test.csv"))
correct = test_df[["rent"]].sort_index()
correct.to_csv(Path("../../Submit/correct.csv"))
# %%
wardlayout = (
    df_unique[["rent", "room_layout", "ward"]]
    .groupby(["room_layout", "ward"])
    .median()
    .reset_index()
)
display(wardlayout)
rough_predict = (
    test_df
    .sort_index()
    .reset_index()
    .merge(
        wardlayout,
        on=["room_layout", "ward"],
        how="left"
    )
    .set_index("index")
)[["rent_x", "rent_y"]]
display(rough_predict.head(10))
predict_df = rough_predict[["rent_y"]].rename(
    columns={
        "rent_y": "rent",
    }
)
predict_df.to_csv(Path("../../Submit/rough_submit.csv"))
predict_df.to_csv(Path("../rough_submit.csv"))
# %%
