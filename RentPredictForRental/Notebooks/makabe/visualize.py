# %%
# Standard lib
import re
import json
from pathlib import Path
from collections import Counter
# Third party
from IPython.display import display
import pandas as pd
import numpy as np
from scipy import stats
from plotly import express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

train_path = Path(r"..\..\Dataset\train.csv")
test_path = Path(r"..\..\Dataset\test.csv")
geojson_path = Path(r"..\\..\\Dataset\tokyo23wards.geojson")
ward_index = {
    '台東区': '13106',
    '杉並区': '13115',
    '文京区': '13105',
    '新宿区': '13104',
    '世田谷区': '13112',
    '江戸川区': '13123',
    '墨田区': '13107',
    '港区': '13103',
    '豊島区': '13116',
    '江東区': '13108',
    '荒川区': '13118',
    '練馬区': '13120',
    '葛飾区': '13122',
    '品川区': '13109',
    '大田区': '13111',
    '中野区': '13114',
    '中央区': '13102',
    '北区': '13117',
    '千代田区': '13101',
    '目黒区': '13110',
    '渋谷区': '13113',
    '板橋区': '13119',
    '足立区': '13121'
}
# %%
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


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


vis_missing(train_data, "train")
vis_missing(test_data, "test")
duplicated_col = [
    'rent',
    'room_layout',
    'floor_area',
    'facing',
    'type',
    'age_of_construction',
    'train',
    'location',
    'option',
    'room_layout_detail',
    'structure',
    'floor',
    'year_of_construction',
    'parking',
    'conditions',
    'total_rooms',
    # 'neighborhood_information',
]
print(train_data.duplicated(subset=duplicated_col, keep=False).sum())
# %%


def split_info(text: str):
    if text != text:
        return ""
    distance = re.findall(split_word, text)
    target = [i.strip() for i in re.split(split_word, text)][0:-1]
    return "\n".join([i + j.strip() for (i, j) in zip(target, distance)])


def get_category(text: str):
    if text == "":
        return ""
    ret = []
    for sentence in text.split("\n"):
        ahead = sentence.count("（")
        rear = sentence.count("）")
        if ahead > 1 and ahead != rear:
            sentence = sentence.split("（", ahead - 1)[-1]
        cate = re.findall(category, sentence)
        if len(cate) == 0:
            continue
        ret.append(cate[-1])
    return " ".join(ret)


pattern = r'都(.*?)区'
split_word = r'\d+m'
category = r'（(.*?)）'
for data in (train_data, test_data):
    data["wards"] = data["location"].apply(
        lambda x: re.search(pattern, x).group(1) + "区"
    )
    data["age"] = data["age_of_construction"].apply(
        lambda x: "0" if x == "新築" else x.strip("築").rstrip("年").rstrip("年以上")
    ).astype(int)
    data["info"] = data["neighborhood_information"].apply(split_info)
    data["info_cate"] = data["info"].apply(get_category)
    data["miss_parking"] = data["parking"].isna().astype(int)
    data["basement"] = data["floor"].apply(
        lambda x: "地下" in x if x == x else False
    ).astype(int)


display(train_data.head(5))
display(test_data.head(5))
# %% [markdown]
# ```
# import geopandas as gpd
# TARGET_WARD = [
#    '千代田区',
#    '中央区',
#    '港区',
#    '新宿区',
#    '文京区',
#    '台東区',
#    '墨田区',
#    '江東区',
#    '品川区',
#    '目黒区',
#    '大田区',
#    '世田谷区',
#    '渋谷区',
#    '中野区',
#    '杉並区',
#    '豊島区',
#    '北区',
#    '荒川区',
#    '板橋区',
#    '練馬区',
#    '足立区',
#    '葛飾区',
#    '江戸川区',
# ]
# DATA_FILE_NAME = "N03-19_13_190101.geojson"
# gdf = gpd.read_file(DATA_FILE_NAME)
# gdf = gdf.query(f"N03_004 in {TARGET_WARD}")
# grouped_gdf = gdf.dissolve(by=["N03_004"],
#                           as_index=False).sort_values("N03_007")
# grouped_gdf.to_file("tokyo23words.geojson", driver="GeoJSON")
# ```
# %%
# https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-v2_3.html#prefecture15
with open(geojson_path, "r", encoding='utf-8') as f:
    tokyo_wards = json.load(f)

df = (
    train_data[["wards", "rent"]]
    .groupby("wards")
    .median()
    .reset_index()
)
wards_index = pd.DataFrame.from_dict(
    ward_index, orient="index", columns=["ward_id"]
)
df = df.merge(wards_index, left_on="wards", right_index=True)
fig = px.choropleth_mapbox(
    data_frame=df,
    geojson=tokyo_wards,
    featureidkey='properties.N03_007',
    hover_name="wards",
    locations="ward_id",
    color="rent",
    color_continuous_scale="Viridis",
    range_color=(70_000, 200_000),
    mapbox_style='carto-positron',
    zoom=10,
    center={"lat": 35.68603008198996, "lon": 139.7529412680478},
    opacity=0.5,
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
fig.show()
fig.write_image("choropleth_map.svg")
# %%
# wards


def create_sunburst(paths: list, title: str):
    fig = px.sunburst(
        train_data,
        path=paths
    )
    fig.update_layout(
        title={
            "text": f"Sunburst {title}",
            "font": {"size": 22, "color": "black"},
            "x": 0.05,
            "y": 0.95,
        },
        margin_l=10,
        margin_b=10,
        margin_t=10,
        height=600,
    )
    fig.show()


create_sunburst(["wards", "room_layout"], "layout")
# create_sunburst(["wards", "structure"], "structure")
# create_sunburst(["wards", "type", "room_layout"], "type layout")
# %%


def plot_scatter(data: pd.DataFrame, x: str):
    fig = px.scatter(
        data,
        x=x,
        y='rent_mean',
        color="wards",
        size='count'
    )
    fig.update_layout(
        margin_l=10,
        margin_b=10,
        margin_t=10,
        height=450,
    )
    fig.show()


df = (
    train_data[["room_layout", "wards", "rent", "floor_area"]]
    .groupby(["room_layout", "wards"])
    .aggregate({"rent": ["mean"], "floor_area": ["mean", "count"]})
)
df.columns = ['rent_mean', 'area_mean', 'count']
df.reset_index(inplace=True)
df["count"] = np.log(df["count"])
plot_scatter(df, "area_mean")
res = []
reg_X = np.arange(0, 1000, 50)
for ward in train_data["wards"].unique():
    ret = pd.DataFrame({"area_mean": reg_X})
    data = df.query(f"wards=='{ward}'")
    model_lr = LinearRegression()
    model_lr.fit(data[['area_mean']], data[['rent_mean']])
    ret["y"] = model_lr.predict(ret)
    ret["ward"] = ward
    res.append(ret)
lr_res = pd.concat(res)
fig = px.line(lr_res, x="area_mean", y="y", color="ward")
fig.update_layout(
    margin_l=10,
    margin_b=10,
    margin_t=10,
    height=450,
)
fig.show()
# ward_dict = {
#    '千代田区': "high",
#    '中央区': "middle",
#    '港区': "high",
#    '新宿区': "lower",
#    '文京区': "middle",
#    '台東区': "middle",
#    '墨田区': "bottom",
#    '江東区': "upper",
#    '品川区': "upper",
#    '目黒区': "middle",
#    '大田区': "bottom",
#    '世田谷区': "lower",
#    '渋谷区': "high",
#    '中野区': "lower",
#    '杉並区': "bottom",
#    '豊島区': "middle",
#    '北区': "bottom",
#    '荒川区': "bottom",
#    '板橋区': "bottom",
#    '練馬区': "bottom",
#    '足立区': "bottom",
#    '葛飾区': "bottom",
#    '江戸川区': "bottom",
# }
# %%
# rent histogram


def rent_histogram(data: pd.DataFrame, title: str, target: str = "rent"):
    fig = px.histogram(
        data,
        x=target,
        nbins=int(data["rent"].max() / 1000),
        marginal="box"
    )
    fig.update_layout(
        title={
            "text": f"{title} for rent",
            "font": {"size": 22, "color": "black"},
            "x": 0.80,
            "y": 0.95,
        },
        margin_l=10,
        margin_b=10,
        margin_t=30,
        height=450,
    )
    fig.show()


rent_histogram(train_data, "all")
train_data["log_rent"] = train_data["rent"].apply(np.log)
rent_histogram(train_data, "log", target="log_rent")
# Boxcox
ret, a = stats.boxcox(train_data["rent"])
train_data["boxcox_rent"] = ret
print(a)
rent_histogram(train_data, "boxcox", target="boxcox_rent")

for ward in train_data["wards"].unique():
    data = train_data.query(f"wards=='{ward}'")
    rent_histogram(data, ward)
    break
# %%
# option conditions,information


def create_wordcloud(words: list):
    wordcloud = WordCloud(
        background_color="white",
        width=1200,
        height=800,
        stopwords=[],
        collocations=False,
        font_path=r"C:\\Windows\\Fonts\\yumin.ttf"
    ).generate(" ".join(words))
    plt.figure(figsize=(12, 9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# option
words = train_data["option"].apply(
    lambda x: x.replace("、", " ") if x == x else ""
)
create_wordcloud(words)
# conditions
words = train_data["conditions"].apply(
    lambda x: x.replace("/", " ") if x == x else ""
)
create_wordcloud(words)
# info_cate
words = train_data["info_cate"]
target = " ".join(words).split(" ")
info_all = Counter(target).most_common()
display(pd.DataFrame(info_all))
create_wordcloud(words)
# %%
# option words
words = train_data["option"].apply(
    lambda x: x.replace("、", " ") if x == x else ""
)
words = " ".join(words).split(" ")
option_items = Counter(words).most_common()
option_items = pd.DataFrame(option_items, columns=["items", "counts"])
option_items = option_items.query("counts>=10")
result_list = []
for it in tqdm(option_items["items"].unique()):
    data = train_data[["rent", "option"]]
    df = train_data["option"].apply(
        lambda x: it in x if x == x and x != "" else False
    )
    df.name = it
    data = pd.concat([data, df], axis=1)
    res_mean = (
        data[["rent", it]]
        .groupby(it)
        .mean()
        .transpose()
    )
    res_mean.columns = [f"{col}_mean" for col in res_mean.columns]
    res_count = (
        data[["rent", it]]
        .groupby(it)
        .count()
        .transpose()
    )
    res_count.columns = [f"{col}_count" for col in res_count.columns]
    res_var = (
        data[["rent", it]]
        .groupby(it)
        .var(ddof=1, numeric_only=False)
        .transpose()
    )
    res_var.columns = [f"{col}_var" for col in res_var.columns]
    res = pd.concat([res_mean, res_count, res_var], axis=1)
    res.index = [it]
    result_list.append(res)

option_agg = pd.concat(result_list)
# %%


def t_test(x):
    mean_1 = x["False_mean"]
    mean_2 = x["True_mean"]
    count_1 = x["False_count"]
    count_2 = x["True_count"]
    var_1 = x["False_var"]
    var_2 = x["True_var"]
    nu = (
        np.power(var_1 / count_1 + var_2 / count_2, 2) /
        (
            np.power(var_1 / count_1, 2) / (count_1 - 1)
            + np.power(var_2 / count_2, 2) / (count_2 - 1)
        )
    )
    t = (mean_1 - mean_2) / np.sqrt(var_1 / count_1 + var_2 / count_2)
    return 1 - stats.t.cdf(x=np.abs(t), df=nu)


option_agg["distance"] = option_agg["True_mean"] - option_agg["False_mean"]
option_agg["p_value"] = option_agg.apply(t_test, axis=1)
option_list = option_agg.query("p_value<0.003 and True_count>=1000")
display(option_list.sort_values("p_value").tail(5))
fig = px.histogram(option_list, x="distance", nbins=80)
fig.show()
df = option_list["distance"].apply(
    lambda x: "nega" if x < 0 else "posi" if 0 < x < 80_000 else "outer"
)
option_dict = dict(zip(option_list.index, df))
print(dict(sorted(option_dict.items(), key=lambda x: x[-1])))
# %%
# conditions
words = train_data["conditions"].fillna("")
words = r"/".join(words).split(r"/")
condition_items = Counter(words).most_common()
condition_items = pd.DataFrame(condition_items, columns=["items", "counts"])
free_rent = condition_items.query("items.str.contains('フリーレント')")
one = pd.DataFrame(
    {
        "items": ["フリーレント1ヶ月"],
        "counts": [
            free_rent.query(
                "items.str.contains('フリーレント1ヶ月')"
            )["counts"].sum()
        ]
    }
)
two = pd.DataFrame(
    {
        "items": ["フリーレント2ヶ月"],
        "counts": [
            free_rent.query(
                "items.str.contains('フリーレント2ヶ月')"
            )["counts"].sum()
        ]
    }
)
three = pd.DataFrame(
    {
        "items": ["フリーレント3ヶ月"],
        "counts": [
            free_rent.query(
                "items.str.contains('フリーレント3ヶ月')"
            )["counts"].sum()
        ]
    }
)
other = pd.DataFrame(
    {
        "items": ["フリーレント"],
        "counts": [
            free_rent[~free_rent["items"].str.contains(
                'フリーレント1ヶ月|フリーレント2ヶ月|フリーレント3ヶ月'
            )]["counts"].sum()
        ]
    }
)
condition_items = pd.concat([
    condition_items[~condition_items["items"].str.contains('フリーレント')],
    one, two, three, other
], axis=0)
condition_items = condition_items.query("counts>=100").reset_index(drop=True)
display(condition_items)
# %%
result_list = []
for it in tqdm(condition_items["items"].unique()):
    data = train_data[["rent", "conditions"]]
    df = train_data["conditions"].apply(
        lambda x: it in x if x == x and x != "" else False
    )
    df.name = it
    data = pd.concat([data, df], axis=1)
    res_mean = (
        data[["rent", it]]
        .groupby(it)
        .mean()
        .transpose()
    )
    res_mean.columns = [f"{col}_mean" for col in res_mean.columns]
    res_count = (
        data[["rent", it]]
        .groupby(it)
        .count()
        .transpose()
    )
    res_count.columns = [f"{col}_count" for col in res_count.columns]
    res_var = (
        data[["rent", it]]
        .groupby(it)
        .var(ddof=1, numeric_only=False)
        .transpose()
    )
    res_var.columns = [f"{col}_var" for col in res_var.columns]
    res = pd.concat([res_mean, res_count, res_var], axis=1)
    res.index = [it]
    result_list.append(res)

condition_agg = pd.concat(result_list)
condition_agg["distance"] = condition_agg["True_mean"] - \
    condition_agg["False_mean"]
condition_agg["p_value"] = condition_agg.apply(t_test, axis=1)
# %%
condition_list = condition_agg.query("p_value<0.025 and True_count>=100")
display(condition_list.sort_values("p_value").tail(5))
fig = px.histogram(condition_list, x="distance", nbins=80)
fig.show()
df = condition_list["distance"].apply(
    lambda x: "nega" if x < -40_000 else "neu"
    if - 10_000 < x < 20_000 else "posi"
)
condition_dict = dict(zip(condition_list.index, df))
print(dict(sorted(condition_dict.items(), key=lambda x: x[-1])))
# %%


def railroad_line(x: str):
    if x != x:
        return 0
    times = [re.search(pattern, it) for it in x.split("\n")]
    times = [int(it.group(1)) for it in times if it is not None]
    times = [it for it in times if it <= 15]
    return len(times)


pattern = r'歩(\d*?)分'
stations = pd.concat(
    [
        train_data[["location", "train"]].copy(),
        test_data[["location", "train"]].copy()
    ]
)
stations["line_num"] = stations["train"].apply(railroad_line)
station_list = (
    stations
    .sort_values("line_num", ascending=False)
    .groupby("location")
    .first()
)
miss_station = pd.concat(
    [
        train_data[train_data["train"].isna()][["location", "train"]].copy(),
        test_data[test_data["train"].isna()][["location", "train"]].copy()
    ]
)
station_list = miss_station.merge(station_list, how="inner", on="location")
print(dict(zip(station_list["location"], station_list["line_num"])))
# %%


def plot_box(data: pd.DataFrame, target: str, title: str):
    fig = px.box(data, x=target, y="rent")
    fig.update_layout(
        title={
            "text": f"{title} relate",
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
        margin={"l": 5, "b": 10},
    )
    fig.show()


plot_box(train_data, "type", "type")
plot_box(train_data, "structure", "structure")
plot_box(train_data, "miss_parking", "miss_parking")
plot_box(train_data, "basement", "basement")
# %%


def get_topmost(x: str) -> int:
    if x == "平屋":
        return 1
    elif r"/" not in x:
        return int(x.rstrip("階建"))

    _, res = x.split(r"/")
    res = res.split("地上")[-1].rstrip("階建")

    return int(res)


def get_floor(x: str) -> int:
    if x == "平屋":
        return 1
    elif r"/" not in x:
        return int(x.rstrip("階建"))

    res, _ = x.split(r"/")
    res = res.rstrip("階")

    if all((res.isdigit(), res.isascii())):
        return int(res)
    elif "-" in res:
        res = [it for it in res.split("-") if it.isdigit()]
        return int(max(res))
    elif "B" in res:
        res = res.strip("B")
        return int(res) if res.isdigit() else 0
    else:
        return 0


age_list = (
    train_data["age_of_construction"]
    .apply(
        lambda x: "0" if x == "新築" else x.strip("築").rstrip("年").rstrip("年以上")
    )
    .astype(int)
)
rooms = (
    train_data["total_rooms"]
    .fillna("0")
    .apply(lambda x: x.rstrip("戸"))
    .astype(int)
)
floor_levels = train_data["floor"].apply(get_floor)
floor_levels.name = "floor_levels"
topmost = train_data["floor"].apply(get_topmost)
topmost.name = "topmost"
numeric_df = pd.concat(
    [train_data["rent"], age_list, rooms, floor_levels, topmost],
    axis=1
)
display(numeric_df)
# %%


def plot_mean_scatter(data: pd.DataFrame, x: str, log: bool = False):
    df = (
        data[["rent", x]]
        .groupby(x)
        .aggregate(["mean", "count"])
    )
    df.columns = ['rent_mean', 'count']
    df["count"] = np.log(df["count"])
    df.reset_index(inplace=True)
    if log:
        df[x] = np.log(df[x] + 1)
    fig = px.scatter(
        df,
        x=x,
        y='rent_mean',
        size='count'
    )
    fig.update_layout(
        margin_l=10,
        margin_b=10,
        margin_t=10,
        height=450,
        title={
            "text": f"{x} corr" if not log else f"{x} corr (log:True)",
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
    )
    fig.show()


plot_mean_scatter(numeric_df, "age_of_construction", log=True)
plot_mean_scatter(numeric_df, "total_rooms")
plot_mean_scatter(numeric_df, "floor_levels")
plot_mean_scatter(numeric_df, "topmost")
# %%
