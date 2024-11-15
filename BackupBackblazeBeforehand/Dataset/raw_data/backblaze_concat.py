# %%
# https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display
from tqdm.notebook import tqdm
# %%
SAMPLING_NUM = 24
columns = [
    'date',
    'serial_number',
    'failure',
]
dtypes = {
    'failure': 'Int8',
}
root = Path(r"backblaze")
files = [file for folder in root.glob("*") for file in folder.glob("*.csv")]
ret = []
for file in tqdm(files):
    df = pd.read_csv(file, usecols=columns + ["model"], dtype=dtypes)
    df = df.query("model!='DELLBOSS VD'")
    df["date"] = pd.to_datetime(df["date"])
    ret.append(df[columns])

serial_data = pd.concat(ret)
display(serial_data.info())
del ret
# %%
# n:11351
# f:302353


def sampling_record(df, record):
    target = [  # noqa
        record["date"].values[0] - np.timedelta64(x, "W")
        for x in range(SAMPLING_NUM)
    ]
    ret = (
        df
        .query("date in @target")
        .sort_values("date")
        .copy()
    )

    return ret


fail_serial = (  # noqa
    serial_data
    .query("failure==1")["serial_number"]
    .unique()
)
# %%
group = (
    serial_data
    .query("serial_number in @fail_serial")
    .groupby("serial_number")
)
failure_list = []
for dataset in tqdm(group.__iter__()):
    df = dataset[1]
    fail_date = df.query("failure==1")["date"].min()
    latest_date = fail_date - np.timedelta64(30, "D")
    if len(latest_record := df.query("@latest_date<date<@fail_date")) == 0:
        continue
    latest_record = latest_record.sample()
    ret = sampling_record(df, latest_record)
    ret["rand"] = np.random.rand()
    ret["failure"] = 1
    failure_list.append(ret)

failure_data = pd.concat(failure_list).drop_duplicates()
del failure_list
# %%
group = (
    serial_data
    .query("serial_number not in @fail_serial")
    .groupby("serial_number")
)
normal_list = []
for dataset in tqdm(group.__iter__()):
    df = dataset[1]
    min_date = (
        df
        .query("failure==0")["date"]
        .min()
    ) + np.timedelta64(SAMPLING_NUM + 1, "W")
    if len(latest_record := df.query("@min_date<date")) == 0:
        continue
    latest_record = latest_record.sample()
    ret = sampling_record(df, latest_record)
    ret["rand"] = np.random.rand()
    normal_list.append(ret)

normal_data = pd.concat(normal_list).drop_duplicates()
del normal_list
# %%
print(len(failure_data["serial_number"].unique()))
print(len(normal_data["serial_number"].unique()))
# 11293/11351
# 269253/291037
# %%
gr = failure_data.groupby("serial_number").aggregate("count")[["date"]]
gr.columns = ["count"]
f_df = pd.merge(
    failure_data,
    gr,
    how="left",
    left_on="serial_number",
    right_index=True
).query("count==@SAMPLING_NUM")
print(len(f_df["serial_number"].unique()))
# 8665
# %%
gr = normal_data.groupby("serial_number").aggregate("count")[["date"]]
gr.columns = ["count"]
n_df = pd.merge(
    normal_data,
    gr,
    how="left",
    left_on="serial_number",
    right_index=True
).query("count==@SAMPLING_NUM")
n_df = n_df.query("0.25<=rand<=0.75")
print(len(n_df["serial_number"].unique()))
# 122270
# %%


def create_data(data):
    rand = pd.DataFrame(
        data["serial_number"].unique(),
        columns=["ser_num"]
    )
    rand["random"] = [np.random.rand() for _ in range(len(rand))]
    ret = pd.merge(
        data,
        rand,
        how="left",
        left_on="serial_number",
        right_on="ser_num"
    ).sort_values("random")[columns]
    display(ret)

    return ret


tr_data = pd.concat(
    [f_df.query("0.23<rand"), n_df.query("0.3<rand")]
)
te_data = pd.concat(
    [f_df.query("rand<0.23"), n_df.query("rand<0.3")]
)
train_data = create_data(tr_data)
test_data = create_data(te_data)
train_data.to_csv("train_sampling.csv", index=False)
test_data.to_csv("test_sampling.csv", index=False)
print(len(train_data.query("failure==0")), len(train_data.query("failure==1")))
print(len(test_data.query("failure==0")), len(test_data.query("failure==1")))
# 2805360 rows × 3 columns
# 337080 rows × 3 columns
