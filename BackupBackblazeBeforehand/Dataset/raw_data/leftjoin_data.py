# %%
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display
from tqdm.notebook import tqdm
# %%
root = Path(r"backblaze")
files = [file for folder in root.glob("*") for file in folder.glob("*.csv")]
rename_col = ["index", "date_smp", "serial_number_smp", "failure_smp"]
train_data = pd.read_csv(Path(r"train_sampling.csv")).reset_index()
test_data = pd.read_csv(Path(r"test_sampling.csv")).reset_index()
train_data["date"] = pd.to_datetime(train_data["date"])
test_data["date"] = pd.to_datetime(test_data["date"])
train_data.columns = rename_col
test_data.columns = rename_col
display(train_data)
display(test_data)
use_col = [
    'date',
    'serial_number',
    'model',
    'capacity_bytes',
    'smart_1_raw',
    'smart_2_raw',
    'smart_3_raw',
    'smart_4_raw',
    'smart_5_raw',
    'smart_7_raw',
    'smart_8_raw',
    'smart_9_raw',
    'smart_10_raw',
    'smart_11_raw',
    'smart_12_raw',
    'smart_13_raw',
    'smart_15_raw',
    'smart_16_raw',
    'smart_17_raw',
    'smart_22_raw',
    'smart_23_raw',
    'smart_24_raw',
    'smart_168_raw',
    'smart_170_raw',
    'smart_173_raw',
    'smart_174_raw',
    'smart_177_raw',
    'smart_179_raw',
    'smart_181_raw',
    'smart_182_raw',
    'smart_183_raw',
    'smart_184_raw',
    'smart_187_raw',
    'smart_188_raw',
    'smart_189_raw',
    'smart_190_raw',
    'smart_191_raw',
    'smart_192_raw',
    'smart_193_raw',
    'smart_194_raw',
    'smart_195_raw',
    'smart_196_raw',
    'smart_197_raw',
    'smart_198_raw',
    'smart_199_raw',
    'smart_200_raw',
    'smart_201_raw',
    'smart_218_raw',
    'smart_220_raw',
    'smart_222_raw',
    'smart_223_raw',
    'smart_224_raw',
    'smart_225_raw',
    'smart_226_raw',
    'smart_231_raw',
    'smart_232_raw',
    'smart_233_raw',
    'smart_235_raw',
    'smart_240_raw',
    'smart_241_raw',
    'smart_242_raw',
    'smart_250_raw',
    'smart_251_raw',
    'smart_252_raw',
    'smart_254_raw',
    'smart_255_raw'
]
# %%
train_ret = []
test_ret = []
for file in tqdm(files):
    df = pd.read_csv(file, usecols=use_col)
    df["date"] = pd.to_datetime(df["date"])
    train = pd.merge(
        train_data,
        df,
        how="inner",
        left_on=["date_smp", "serial_number_smp"],
        right_on=["date", "serial_number"],
    )
    train_ret.append(train)
    test = pd.merge(
        test_data,
        df,
        how="inner",
        left_on=["date_smp", "serial_number_smp"],
        right_on=["date", "serial_number"],
    )
    test_ret.append(test)
train = pd.concat(train_ret).drop_duplicates(subset=rename_col)
test = pd.concat(test_ret).drop_duplicates(subset=rename_col)
print(len(train), len(test))
# %%


def sort_date(data):
    res = (
        data
        .groupby('serial_number_smp')
        .apply(lambda x: x.sort_values(['date_smp'], ascending=True))
        .reset_index(drop=True)
    )
    res.rename(columns={"failure_smp": "failure"}, inplace=True)
    data_col = use_col + ["failure"]

    return res[data_col]


sort_date(train).to_csv("train.csv", index=False)
test_data = sort_date(test)
correct_data = (
    test_data[["serial_number", "failure"]]
    .groupby("serial_number")
    .apply(np.prod, axis=0)
    .reset_index()
)[["serial_number", "failure"]]
correct_data.to_csv("correct.csv", index=False)
test_data.drop(["failure"], axis=1).to_csv("test.csv", index=False)

# %%
