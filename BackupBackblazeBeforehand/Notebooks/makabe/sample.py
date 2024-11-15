# %%
from pathlib import Path
from IPython.display import display
import pandas as pd

DATA_PATH = Path(r"..\..\Dataset\test.csv")
CORRECT_PATH = Path(r"..\..\.github\workflows\correct.csv")
# %%
row_df = pd.read_csv(DATA_PATH)
correct = pd.read_csv(CORRECT_PATH)
correct.columns = ["serial_number", "FF"]
gr_df = row_df[["serial_number"]].groupby("serial_number").count()
gr_df["failure"] = 0
df = pd.merge(correct, gr_df, on="serial_number", how="left").sample(frac=1)
display(df)
corr_df = df[["serial_number", "FF"]]
corr_df.columns = ["serial_number", "failure"]
corr_df.to_csv(CORRECT_PATH.parent / "correct_submit.csv", index=False)
sample_df = df[["serial_number", "failure"]]
sample_df.to_csv(DATA_PATH.parent / "sample_submit.csv", index=False)
# %%
