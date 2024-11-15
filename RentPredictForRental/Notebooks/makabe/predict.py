# %%
# standard lib
import re
from pathlib import Path
from dataclasses import dataclass, field
# third party
from IPython.display import display
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb

PARAM_SEARCH = False
TRAIN_PATH = Path(r"..\..\Dataset\train.csv")
TEST_PATH = Path(r"..\..\Dataset\test.csv")
SUBMIT_PATH = Path(r"..\..\Submit\rough_submit.csv")


@dataclass
class ConstData:
    options: dict = field(default_factory=lambda: {
        'エアコン': 'nega', '都市ガス': 'nega', '駅徒歩10分以内': 'nega',
        '2口コンロ': 'nega', '2沿線利用可': 'nega', '初期費用カード決済可': 'nega',
        '即入居可': 'nega', '光ファイバー': 'nega', '2駅利用可': 'nega',
        'IT重説': 'nega', '対応物件': 'nega', '礼金不要': 'nega',
        '保証金不要': 'nega', '敷金不要': 'nega', '高速ネット対応': 'nega',
        '外壁タイル張り': 'nega', '陽当り良好': 'nega', '照明付': 'nega',
        '閑静な住宅地': 'nega', '最上階': 'nega', '敷金・礼金不要': 'nega',
        'IHクッキングヒーター': 'nega', '駅まで平坦': 'nega', '単身者相談': 'nega',
        '全居室6畳以上': 'nega', '平坦地': 'nega', '上階無し': 'nega',
        '室内物干機': 'nega', '年内入居可': 'nega', 'CATVインターネット': 'nega',
        '始発駅': 'nega', '押入': 'nega', 'ロフト': 'nega',
        '仲手0.55ヶ月': 'nega', 'キッチンに窓': 'nega', 'ルームシェア相談': 'nega',
        '年度内入居可': 'nega', 'シャッター': 'nega', '家賃カード決済可': 'nega',
        '学生相談': 'nega', 'プロパンガス': 'nega', '和室': 'nega',
        '内装リフォーム済': 'nega', '緑豊かな住宅地': 'nega', '出窓': 'nega',
        '冷蔵庫': 'nega', '防犯シャッター': 'nega', '全室照明付': 'nega',
        '家具付': 'nega', '家電付': 'nega', '当社管理物件': 'nega',
        '雨戸': 'nega', 'バス停徒歩3分以内': 'nega', '床下収納': 'nega',
        'シャワールーム': 'nega', '電気コンロ': 'nega', 'クッションフロア': 'nega',
        '全室2面採光': 'nega', 'LGBTフレンドリー': 'nega', '1フロア2住戸': 'nega',
        '電子キー': 'nega', 'テレビ': 'nega', '電子レンジ': 'nega',
        '内階段': 'nega', '収納1間半': 'nega', 'カーテン付': 'nega',
        'ベッド': 'nega', '天袋': 'nega', '高齢者歓迎': 'nega',
        '電子ロック': 'nega', 'コインランドリー': 'nega', '初期費用30万円以下': 'nega',
        '前面棟無': 'nega', '外壁サイディング': 'nega', '2×4工法': 'nega',
        'CATV使用料不要': 'nega', '初期費用20万円以下': 'nega', '内装コンクリート': 'nega',
        '内装リフォーム後渡': 'nega', '準耐火構造': 'nega', '階段下収納': 'nega',
        'フローリング風フロアタイル': 'nega', '初期費用15万円以下': 'nega', 'ダイニングテーブル・デスク': 'nega',
        '室内洗濯置': 'posi', 'バストイレ別': 'posi', 'フローリング': 'posi',
        'シューズボックス': 'posi', 'バルコニー': 'posi', 'TVインターホン': 'posi',
        'システムキッチン': 'posi', 'クロゼット': 'posi', 'オートロック': 'posi',
        '温水洗浄便座': 'posi', '洗面所独立': 'posi', '駐輪場': 'posi',
        'ガスコンロ対応': 'posi', '浴室乾燥機': 'posi', '宅配ボックス': 'posi',
        '3駅以上利用可': 'posi', 'エレベーター': 'posi', '敷地内ごみ置き場': 'posi',
        '防犯カメラ': 'posi', '角住戸': 'posi', '敷金1ヶ月': 'posi',
        '洗面化粧台': 'posi', '脱衣所': 'posi', '追焚機能浴室': 'posi',
        'CATV': 'posi', '3沿線以上利用可': 'posi', '全居室フローリング': 'posi',
        'BS': 'posi', '駅徒歩5分以内': 'posi', '礼金1ヶ月': 'posi',
        'ネット使用料不要': 'posi', '24時間換気システム': 'posi', 'BS・CS': 'posi',
        '2面採光': 'posi', 'CS': 'posi', '24時間ゴミ出し可': 'posi',
        'ペット相談': 'posi', '南向き': 'posi', '洗面所にドア': 'posi',
        '全居室洋室': 'posi', 'ディンプルキー': 'posi', '二人入居相談': 'posi',
        '通風良好': 'posi', 'バイク置場': 'posi', 'シャワー付洗面台': 'posi',
        'グリル付': 'posi', 'ネット専用回線': 'posi', 'ダブルロックキー': 'posi',
        '分譲賃貸': 'posi', '南面バルコニー': 'posi', '玄関収納': 'posi',
        'オートバス': 'posi', 'エアコン全室': 'posi', '耐震構造': 'posi',
        '全居室収納': 'posi', '築2年以内': 'posi', 'デザイナーズ': 'posi',
        '眺望良好': 'posi', 'ガスレンジ付': 'posi', '玄関ホール': 'posi',
        '築5年以内': 'posi', '24時間緊急通報システム': 'posi', '南西向き': 'posi',
        '東南向き': 'posi', 'フリーレント': 'posi', 'エアコン2台': 'posi',
        '耐火構造': 'posi', '仲介手数料不要': 'posi', '南面リビング': 'posi',
        '人感照明センサー': 'posi', '築3年以内': 'posi', '日勤管理': 'posi',
        'クロゼット2ヶ所': 'posi', '独立型キッチン': 'posi', '複層ガラス': 'posi',
        '振分': 'posi', 'セキュリティ会社加入済': 'posi', 'トイレ未使用': 'posi',
        'マルチメディアコンセント': 'posi', 'LAN': 'posi', '浴室に窓': 'posi',
        '一部フローリング': 'posi', '東南角住戸': 'posi', '駅前': 'posi',
        '3面採光': 'posi', '南西角住戸': 'posi', '3方角住戸': 'posi',
        'ダウンライト': 'posi', '浴室未使用': 'posi', '南面2室': 'posi',
        '全室南向き': 'posi', '事務所相談': 'posi', 'キッチン未使用': 'posi',
        '全居室8畳以上': 'posi', '未入居': 'posi', 'テラス': 'posi',
        '風除室': 'posi', '外壁コンクリート': 'posi', '平面駐車場': 'posi',
        'オール電化': 'posi', '天井高シューズクロゼット': 'posi', '洗濯機': 'posi',
        'ペット専用設備': 'posi', '三面鏡付洗面化粧台': 'posi', '礼金2ヶ月': 'posi',
        'メゾネット': 'posi', '洗面所に窓': 'posi', 'リノベーション': 'posi',
        'ワイドバルコニー': 'posi', '縦型照明付洗面化粧台': 'posi', '防犯ガラス': 'posi',
        '寝室8畳以上': 'posi', '間接照明': 'posi', '免震構造': 'posi',
        '2WAYバルコニー': 'posi', 'オープンキッチン': 'posi', '全室東南向き': 'posi',
        '全室南西向き': 'posi', '1フロア1住戸': 'posi', '可動間仕切り': 'posi',
        '備付食器棚': 'posi', '専用庭': 'posi', '四方角部屋': 'posi',
        'ISDN対応': 'posi', '玄関ポーチ': 'posi', 'オートライト': 'posi',
        '耐風構造': 'posi', '乾燥機': 'posi', '制震構造': 'posi',
        '防音サッシ': 'posi', '浴室1坪以上': 'posi', 'TV付浴室': 'posi',
        '収納2間': 'posi', '高気密住宅': 'posi', '庭': 'posi',
        'バリアフリー': 'posi', '文教地区': 'posi', '高台に立地': 'posi',
        'クリーニングボックス': 'posi', 'リバーサイド': 'posi', '電動シャッター': 'posi',
        '二重床構造': 'posi',
        '3口以上コンロ': 'outer', 'ウォークインクロゼット': 'outer', '対面式キッチン': 'outer',
        'LDK12畳以上': 'outer', '楽器相談': 'outer', '床暖房': 'outer',
        '浄水器': 'outer', '高層階': 'outer', 'シューズWIC': 'outer',
        '食器洗乾燥機': 'outer', 'エレベーター2基': 'outer', 'LDK15畳以上': 'outer',
        '2面バルコニー': 'outer', 'タワー型マンション': 'outer', 'フロントサービス': 'outer',
        '敷金2ヶ月': 'outer', 'トランクルーム': 'outer', 'クロゼット3ヶ所': 'outer',
        'ディスポーザー': 'outer', 'エアコン3台': 'outer', '24時間有人管理': 'outer',
        'ルーフバルコニー': 'outer', '納戸': 'outer', 'L字型キッチン': 'outer',
        'LDK18畳以上': 'outer', 'トイレ2ヶ所': 'outer', 'LDK20畳以上': 'outer',
        'エアコン4台': 'outer', '専有面積25坪以上': 'outer', '自走式駐車場': 'outer',
        '駐車場1台無料': 'outer', 'ミストサウナ': 'outer', 'ウォークインクロゼット2': 'outer',
        'ゲストルーム': 'outer', '両面バルコニー': 'outer', 'タンクレストイレ': 'outer',
        '専有面積30坪以上': 'outer', 'L字型バルコニー': 'outer', '寝室10畳以上': 'outer',
        'ビルトインエアコン': 'outer', '南面3室': 'outer', '駐車2台可': 'outer',
        'LDK25畳以上': 'outer', 'キッズルーム': 'outer', 'オーブンレンジ': 'outer',
        'サウナ': 'outer',
        # '': 'posi'
    })
    conditions: dict = field(default_factory=lambda: {
        '単身者限定': 'nega', '子供不可': 'nega', '女性限定': 'nega',
        '学生希望': 'nega', '学生限定': 'nega', '男性限定': 'nega',
        '単身者可': 'neu', '事務所利用不可': 'neu', 'ルームシェア不可': 'neu',
        'ルームシェア相談': 'neu', '事務所利用相談': 'neu', '法人希望': 'neu',
        '法人限定': 'neu', 'フリーレント1ヶ月': 'neu',
        '二人入居可': 'posi', 'ペット相談': 'posi', '子供可': 'posi',
        '楽器相談': 'posi', 'フリーレント2ヶ月': 'posi', 'フリーレント3ヶ月': 'posi'
        # 'フリーレント': 'neu',
        # '': 'posi',
    })
    miss_walk_stations: dict = field(default_factory=lambda: {
        '東京都港区高輪１': 3,
        '東京都北区滝野川７': 3,
        '東京都墨田区東駒形４': 3,
        '東京都北区赤羽西１': 3,
        '東京都北区田端新町２': 3,
        '東京都荒川区西尾久７': 3,
        '東京都港区浜松町１': 3,
        '東京都江東区亀戸６': 3,
        '東京都豊島区東池袋２': 3,
        '東京都荒川区東日暮里５': 3,
        '東京都品川区上大崎２': 3,
        '東京都港区高輪４': 3,
        '東京都杉並区西荻南３': 3,
        '東京都江戸川区南小岩５': 3,
        '東京都渋谷区恵比寿南２': 3,
        '東京都大田区蒲田５': 3,
        '東京都足立区入谷２': 3,
        '東京都中野区中野１': 3,
        '東京都新宿区下落合３': 3,
        '東京都中野区弥生町５': 3,
        '東京都台東区東上野１': 3,
        '東京都品川区西五反田４': 3,
        '東京都品川区大井６': 3
    })
    miss_close_stations: dict = field(default_factory=lambda: {
        '東京都港区高輪１': 2,
        '東京都北区滝野川７': 2,
        '東京都墨田区東駒形４': 2,
        '東京都北区赤羽西１': 3,
        '東京都北区田端新町２': 1,
        '東京都荒川区西尾久７': 3,
        '東京都港区浜松町１': 3,
        '東京都江東区亀戸６': 2,
        '東京都豊島区東池袋２': 3,
        '東京都荒川区東日暮里５': 3,
        '東京都品川区上大崎２': 3,
        '東京都港区高輪４': 3,
        '東京都杉並区西荻南３': 3,
        '東京都江戸川区南小岩５': 2,
        '東京都渋谷区恵比寿南２': 2,
        '東京都大田区蒲田５': 3,
        '東京都足立区入谷２': 1,
        '東京都中野区中野１': 2,
        '東京都新宿区下落合３': 1,
        '東京都中野区弥生町５': 2,
        '東京都台東区東上野１': 3,
        '東京都品川区西五反田４': 2,
        '東京都品川区大井６': 0
    })
    structure_replace: dict = field(default_factory=lambda: {
        '鉄筋コン': "鉄骨鉄筋コン",
        '鉄骨鉄筋': "鉄骨鉄筋コン",
        '鉄骨': "鉄骨木造",
        '木造': "鉄骨木造",
        'ブロック': "鉄骨木造",
        '気泡コン': "ALCPC",
        'プレコン': "ALCPC",
        '軽量鉄骨': "その他",
        '鉄骨プレ': "その他",
        'その他': "その他",
    })


CD = ConstData()
# %%
# read data
train_data = pd.read_csv(TRAIN_PATH, index_col=0)
test_data = pd.read_csv(TEST_PATH, index_col=0)
# %%
# feature engineering
pattern_ward = r'都(.*?)区'
pattern_station = r'歩(\d*?)分'
pattern_split = r'\d+m'


def railroad_line(x: str, minutes: int) -> int:
    # minutes分以内の路線
    times = [re.search(pattern_station, it) for it in x.split("\n")]
    times = [int(it.group(1)) for it in times if it is not None]
    times = [it for it in times if it <= minutes]
    return len(times)


def count_item(x: str, target: list) -> int:
    if x != x:
        return 0
    split_options = x.split("、")
    ret = set(split_options) & set(target)
    return len(ret)


def count_info(x: str) -> int:
    if x != x:
        return 0
    distance = re.findall(pattern_split, x)
    res = [i for i in distance if int(i.rstrip("m")) <= 1000]
    return len(res)


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


nega_options = [key for key, val in CD.options.items() if val == "nega"]
posi_options = [key for key, val in CD.options.items() if val == "posi"]
outer_options = [key for key, val in CD.options.items() if val == "outer"]
nega_conditions = [key for key, val in CD.conditions.items() if val == "nega"]
posi_conditions = [key for key, val in CD.conditions.items() if val == "posi"]
for data in (train_data, test_data):
    data["ward"] = data["location"].apply(
        lambda x: re.search(pattern_ward, x).group(1) + "区"
    )
    data["age"] = data["age_of_construction"].apply(
        lambda x: "0" if x == "新築" else x.strip("築").rstrip("年").rstrip("年以上")
    ).astype(int)
    data["age"] = np.log(data["age"] + 1)
    data["structure"] = data["structure"].replace(CD.structure_replace)
    data["north"] = data["facing"].fillna("").apply(
        lambda x: 1 if "北" in x else 0
    )
    data["neighbors_num"] = data["neighborhood_information"].apply(count_info)
    data["lines_num"] = data[["location", "train"]].apply(
        lambda x: railroad_line(x.iloc[1], 15)
        if x.iloc[1] == x.iloc[1] else CD.miss_walk_stations[x.iloc[0]],
        axis=1,
    )
    data["topmost"] = data["floor"].apply(get_topmost)
    data["basement_flag"] = data["floor"].apply(
        lambda x: "地下" in x
    ).astype(int)
    data["floor_level"] = data["floor"].apply(get_floor)
    data["high_flag"] = data["floor_level"].apply(lambda x: 1 if x > 15 else 0)
    data["tower_flag"] = data["topmost"].apply(lambda x: 1 if x > 15 else 0)
    data["neg_options"] = data["option"].apply(count_item, args=[nega_options])
    data["pos_options"] = data["option"].apply(count_item, args=[posi_options])
    data["outer_options"] = data["option"].apply(
        count_item, args=[outer_options]
    )
    data["neg_conditions"] = data["conditions"].apply(
        count_item, args=[nega_conditions]
    )
    data["pos_conditions"] = data["conditions"].apply(
        count_item, args=[posi_conditions]
    )
    data["miss_facing"] = data["facing"].isna().astype(int)
    data["miss_parking"] = data["parking"].isna().astype(int)
    data["miss_option"] = data["option"].isna().astype(int)
    data["miss_conditions"] = data["conditions"].isna().astype(int)
# %%
# one hot
all_data = pd.concat([train_data, test_data])
all_data = pd.get_dummies(all_data, columns=["type", "structure"], dtype=int)
train_data = all_data.loc[train_data.index]
test_data = all_data.loc[test_data.index]
del all_data
# %%
# linear regression


def reg_pre(x: tuple):
    ward, area = x
    coef, intercept = linearmodel_dict[ward]
    return coef * area + intercept


linearmodel_dict = {}
for ward in tqdm(train_data["ward"].unique()):
    data = train_data.query(f"ward=='{ward}'")
    model_lr = LinearRegression()
    model_lr.fit(data[["floor_area"]], data[["rent"]])
    linearmodel_dict[ward] = (model_lr.coef_[0][0], model_lr.intercept_[0])
for data in (train_data, test_data):
    data["area_rent"] = data[["ward", "floor_area"]].apply(reg_pre, axis=1)
# %%
# preprocess
use_col = [
    'area_rent',
    'type_アパート', 'type_マンション', 'type_一戸建て',
    'type_テラス・タウンハウス', 'type_その他',
    'structure_鉄骨鉄筋コン', 'structure_鉄骨木造', 'structure_ALCPC', 'structure_その他',
    'age', 'north', 'neighbors_num', 'lines_num',
    'topmost', 'basement_flag', 'tower_flag', 'floor_level', 'high_flag',
    'neg_options', 'pos_options', 'outer_options',
    'neg_conditions', 'pos_conditions',
    'miss_facing', 'miss_parking', 'miss_option', 'miss_conditions'
]
numeric = [
    'area_rent',
    'age',
    'neighbors_num',
    'lines_num',
    'topmost',
    'floor_level',
    'neg_options',
    'pos_options',
    'outer_options',
    'neg_conditions',
    'pos_conditions',
]
scaler = StandardScaler()
scaler.fit(train_data[numeric])
for data in (train_data, test_data):
    data[numeric] = pd.DataFrame(
        scaler.transform(data[numeric]),
        columns=numeric,
        index=data.index
    )
target_data = train_data[["rent"]].copy()
train_data = train_data[use_col].copy()
test_data = test_data[use_col].copy()
display(train_data.head(10))
# stats
F_vals, p_vals = f_regression(train_data, target_data.values.ravel())
stats_data = pd.DataFrame(
    {"F_val": F_vals, "p_val": p_vals},
    index=train_data.columns
)
stats_data["VIF"] = pd.Series(
    [
        variance_inflation_factor(train_data.values, i)
        for i in range(len(use_col))
    ],
    index=train_data.columns,
)
# %%
# predict ElasticNet
estimator = ElasticNet
params = {'alpha': 0.4, 'l1_ratio': 0.5}
# score___0.12314


def calc_smape(
    true_value: pd.DataFrame,
    forecast_value: np.array
) -> np.float64:
    """_summary_

    Args:
        true_value (pd.DataFrame): _description_
        forecast_value (np.array): _description_

    Returns:
        np.float64: _description_
    """
    true_value = true_value["rent"]
    score = np.average(
        np.abs(forecast_value - true_value) /
        (forecast_value + true_value) * 2
    )

    return np.round(score, 4)


def model_predict(estimator, param_grid: dict, param: dict = {}):
    scorer = make_scorer(calc_smape, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    match PARAM_SEARCH:
        # tuning and fit-predict
        case True:
            grid = GridSearchCV(
                estimator=estimator(),
                param_grid=param_grid,
                scoring=scorer,
                cv=kf,
                n_jobs=-1
            )
            grid.fit(train_data, target_data)
            print("Best Parameter: ", grid.best_params_)
            model = grid.best_estimator_
            res = (
                np.abs(grid.best_score_),
                np.abs(model.coef_.T),
                model.predict(train_data),
                model.predict(test_data),
            )
        # fit-predict
        case False:
            print("Stable Parameter: ", param)
            score_list = []
            importance = []
            train_predict = []
            test_predict = []
            for fold, (train_idx, validate_idx) in enumerate(
                kf.split(train_data),
                start=1
            ):
                model = estimator(**param)
                model.fit(
                    train_data.iloc[train_idx],
                    target_data.iloc[train_idx].values.ravel()
                )
                score = calc_smape(
                    target_data.iloc[validate_idx],
                    model.predict(train_data.iloc[validate_idx])
                )
                print(fold, score)
                score_list.append(score)
                importance.append(np.abs(model.coef_.T))
                train_predict.append(model.predict(train_data))
                test_predict.append(model.predict(test_data))
            res = (
                np.average(score_list),
                np.average(importance, axis=0),
                np.average(train_predict, axis=0),
                np.average(test_predict, axis=0)
            )

    return res


param_grid = {
    "alpha": np.linspace(0.2, 0.6, 5),
    "l1_ratio": np.linspace(0.3, 0.7, 5),
}
score, importance, train_predict, test_predict = model_predict(
    estimator, param_grid, params
)
# %%


def residual_plot(true_val: pd.DataFrame, predict: pd.DataFrame):
    residual_data = pd.DataFrame(
        {
            "true_val": true_val.values.ravel(),
            "predict_val": predict,
        },
        index=target_data.index
    )
    residual_data["residual"] = (
        residual_data["true_val"]
        - residual_data["predict_val"]
    )
    residual_data = residual_data.query("residual<-30_000 or residual>30_000")
    fig = px.scatter(
        residual_data,
        x="true_val",
        y="residual",
        color="predict_val",
        trendline="ols",
        render_mode='webgl'
    )
    fig.update_layout(
        # title={
        #    "text": f"{title} for rent",
        #    "font": {"size": 22, "color": "black"},
        #    "x": 0.80,
        #    "y": 0.95,
        # },
        margin_l=10,
        margin_b=10,
        margin_t=30,
        height=450,
    )
    fig.show()
    print("residual average", np.average(residual_data["residual"]))


stats_data["importance"] = pd.Series(importance, index=train_data.columns)
display(stats_data.sort_values("importance", ascending=False))
print(f"{estimator}:score___{score}")
residual_plot(target_data, train_predict)
# %%
# to csv


def save_result(predict_data: pd.DataFrame, file_name: str):
    result = pd.DataFrame(
        predict_data,
        index=test_data.index,
        columns=["rent"]
    ).reset_index()
    result.to_csv(
        SUBMIT_PATH.parent / file_name,
        index=False
    )
    display(result)


save_result(test_predict, "result_makabe_Elastic.csv")
# %%
# LightGBM
PARAM_SEARCH = False
params = {
    "device": "gpu",
    "objective": "regression",
    "boosting_type": "gbdt",
    "max_bins": 255,
    "metric": "mae",
    "learning_rate": 0.05,
    "verbose": -1,
}
kf = KFold(n_splits=5, shuffle=True, random_state=37)
# tuning
match PARAM_SEARCH:
    case True:
        train_set = lgb.Dataset(train_data, target_data)
        tuner = opt_lgb.LightGBMTunerCV(
            params,
            train_set=train_set,
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
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'max_bins': 255,
            'metric': 'l2',
            'learning_rate': 0.05,
            'verbose': -1,
            'feature_pre_filter': False,
            'lambda_l1': 4.115085379472325e-07,
            'lambda_l2': 0.001486444084761772,
            'num_leaves': 252,
            'feature_fraction': 0.9159999999999999,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 10
        }
        # SMAPE 0.06914
# fit predict
score_list = []
train_predict = []
test_predict = []
importance = []
for fold, (train_idx, validate_idx) in enumerate(
    kf.split(train_data),
    start=1
):
    lgb_model = lgb.LGBMRegressor(
        **params,
        n_estimators=2000,
    )
    lgb_model.fit(
        train_data.iloc[train_idx],
        target_data.iloc[train_idx],
        eval_set=(
            train_data.iloc[validate_idx],
            target_data.iloc[validate_idx],
        ),
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0),
        ]
    )
    score = calc_smape(
        target_data.iloc[validate_idx],
        lgb_model.predict(train_data.iloc[validate_idx])
    )
    score_list.append(score)
    test_predict.append(lgb_model.predict(test_data))
    train_predict.append(lgb_model.predict(train_data))
    importance.append(lgb_model.feature_importances_)
    print(fold, score)

stats_data["lgb_importance"] = pd.Series(
    np.average(importance, axis=0),
    index=train_data.columns
)
print(params)
print("SMAPE", np.mean(score_list))
display(stats_data.sort_values("lgb_importance", ascending=False))
residual_plot(target_data, np.average(train_predict, axis=0))
save_result(np.average(test_predict, axis=0), "result_makabe_LGBMReg.csv")
# %%
