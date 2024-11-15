import requests
from bs4 import BeautifulSoup
from retry import retry
import pandas as pd
from tqdm import tqdm
import time

START_PAGE = 1
BASE_PAGE = "https://suumo.jp"
# 1000ページ超えると取得できないので区で分割
TARGET_PAGE_DICT = {
    # "千代田区": (
    #    218,
    #    "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
    #    "sc=13101&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
    #    "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    # ),
    "中央区": (
        353,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13102&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "港区": (
        730,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13103&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "新宿区": (
        797,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13104&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "文京区": (
        437,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13105&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "渋谷区": (
        589,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13113&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "台東区": (
        527,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13106&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "墨田区": (
        517,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13107&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "江東区": (
        596,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13108&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "荒川区": (
        193,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13118&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "足立区": (
        588,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13121&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "葛飾区": (
        355,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13122&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "江戸川区": (
        454,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13123&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "品川区": (
        622,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13109&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "目黒区": (
        454,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13110&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "太田区": (
        770,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13111&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    # "世田谷区": (
    #    1077,
    #    "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
    #    "sc=13112&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
    #    "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    # ),
    "中野区": (
        482,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13114&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "杉並区": (
        739,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13115&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "練馬区": (
        667,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13120&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "豊島区": (
        435,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13116&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "北区": (
        493,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13117&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    ),
    "板橋区": (
        636,
        "https://suumo.jp/jj/chintai/ichiran/FR301FC005/?ar=030&bs=040&ta=13&"
        "sc=13119&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shk"
        "r1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99&pc=100"
    )
}


@retry(tries=3, delay=10, backoff=2)
def load_page(url: str) -> BeautifulSoup:
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    return soup


def get_content(soup: BeautifulSoup, tag: str, target: dict) -> str:
    if soup is None:
        return None
    res = soup.find(tag, target)
    if res is None:
        return None
    else:
        return res.get_text(strip=True)


def wrap_find(
        soup: BeautifulSoup,
        find_args: list,
        find_all_args: list
) -> str:
    res = soup.find(*find_args)
    if res is None:
        return None
    else:
        return res.find_all(*find_all_args)


def scraping_text(soup: BeautifulSoup) -> dict:
    data = {}
    property_data = wrap_find(
        soup,
        ["div", {"class": "l-property_view_detail-main"}],
        ["li"]
    )
    property_train = wrap_find(
        soup,
        [
            "div",
            {"class": "property_view_detail property_view_detail--train"}
        ],
        ["div", {"class": "property_view_detail-text"}]
    )
    property_table = soup.find(
        "table",
        {"class": "data_table table_gaiyou"}
    )
    if any((
        property_train is None,
        property_table is None,
        property_data is None,
    )):
        print(property_train, property_table, property_data)
        return False

    # 賃料管理費
    data["rent"] = get_content(
        soup,
        "div",
        {"class": "property_view_main-emphasis"}
    )
    data["fee"] = get_content(
        soup,
        "div",
        {"class": "property_data-body"}
    )

    # 基礎データ
    for item in property_data:
        category = get_content(
            item,
            "div",
            {"class": "property_data-title"}
        )
        text = get_content(item, "div", {"class": "property_data-body"})
        data[category] = text

    # 最寄り、場所、オプション
    data["train"] = "\n".join(
        [item.get_text(strip=True) for item in property_train]
    )
    data["location"] = get_content(
        soup.find(
            "div",
            {"class": "property_view_detail property_view_detail--location"}
        ),
        "div",
        {"class": "property_view_detail-text"}
    )
    data["option"] = get_content(
        soup.find("div", {"id": "bkdt-option"}),
        "ul",
        {"class": "inline_list"}
    )

    # その他情報
    table_column = [
        elem.get_text(strip=True)
        for elem in property_table.find_all("th")
    ]
    table_column = [i for i in table_column if len(i) != 0]
    table_item = [
        elem.get_text(strip=True)
        for elem in property_table.find_all("td")
    ]
    table_item = [i for i in table_item if len(i) != 0]
    for col, it in zip(table_column, table_item):
        if col in [
            '間取り詳細',
            '構造',
            '階建',
            '築年月',
            '駐車場',
            '条件',
            '総戸数',
            '周辺情報',
        ]:
            data[col] = it

    for key, it in data.items():
        if it == "-":
            it = None
        data[key] = it

    return data


def main(division: str, max_page: int, target_page: str):
    result_list = []
    for page in tqdm(range(START_PAGE, max_page + START_PAGE)):
        page_num = f"&page={page}" if page != 1 else ""
        soup = load_page(target_page + page_num)
        articles = [
            elem.find_all("div", {"class": "property_inner"})
            for elem in soup.find_all("div", {"class": "property_group"})
        ]
        articles = [
            elem.find("a").attrs["href"]
            for article in articles for elem in article
        ]

        res = []
        for article in tqdm(articles):
            target = BASE_PAGE + article
            data = scraping_text(load_page(target))
            if data is False:
                continue
            data["URL"] = target
            res.append(data)

        df = pd.DataFrame(res).reset_index(drop=True)
        result_list.append(df)
        df.to_csv("result_suumo.csv", mode="a", header=False)
        time.sleep(1)

    df = pd.concat(result_list).reset_index(drop=True)
    df.to_csv(f"result_suumo_finally_{division}.csv")


if __name__ == "__main__":
    for division, (max_page, target_page) in TARGET_PAGE_DICT.items():
        main(division, max_page, target_page)
