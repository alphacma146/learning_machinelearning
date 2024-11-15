# %%
# standard lib
from pathlib import Path
# third party
import polars as pl
from scipy.signal import butter, filtfilt
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
# %%
train_path = Path(r"..\..\Dataset\train_data.csv")
train_df = pl.read_csv(train_path)
train_df = train_df.with_columns(pl.col("Date").cast(pl.Date))
display(train_df.head(10))

fig = go.Figure()
fig.add_trace(go.Bar(
    x=train_df["Date"], y=train_df["Volume"],
    name="Volume", yaxis="y"
))
fig.add_trace(go.Scatter(
    x=train_df["Date"], y=train_df["Close"],
    name="Date", mode="lines", yaxis="y2"
))
fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis2=dict(title="Close", side="left", overlaying="y"),
    yaxis=dict(title='Volume', side="right", overlaying=None, showgrid=False),
)
fig.show()
# %%
df_rolling = train_df.select([
    train_df["Date"],
    train_df["Close"],
    pl.col("Close").rolling_mean(20).alias("20_rolling"),
    pl.col("Close").rolling_mean(75).alias("75_rolling"),
    pl.col("Close").rolling_mean(120).alias("120_rolling")
])
fig = px.line(
    df_rolling, x="Date", y=[
        "Close", "20_rolling", "75_rolling", "120_rolling"
    ]
)
fig.show()
# %%
df_acor = (
    train_df
    .select(
        [train_df["Close"].shift(i).alias(f"lag_{i}") for i in range(365)]
    )
    .fill_null(0)
    .corr()
    .head(1)
    .transpose(include_header=True, header_name="lag", column_names=["acor"])
)
fig = px.bar(df_acor, x="lag", y="acor")
fig.show()
# fig.write_html("bar_chart.html")
# %%
df_change = (
    train_df
    .select([
        pl.col("Date"),
        pl.col("Close"),
        pl.col("Close").shift(1).alias("prev_val"),
    ])
    .with_columns(
        (
            (pl.col("Close") - pl.col("prev_val")) / pl.col("prev_val")
        ).alias("change_rate")
    )
)
fig = px.bar(df_change, x="Date", y="change_rate")
fig.show()
# %%


def split_signal_noise(data, normal_cutoff, order=5):
    b_low, a_low = butter(order, normal_cutoff, btype='low', analog=False)
    b_high, a_high = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_low = filtfilt(b_low, a_low, data)
    filtered_high = filtfilt(b_high, a_high, data)
    return filtered_low, filtered_high


nc = 0.025
low, high = split_signal_noise(
    train_df.select([pl.col("Close")]).to_numpy().ravel(),
    nc
)
df = pl.DataFrame({'low': low, 'high': high})
df = df.with_columns(
    sum=pl.col("low") + pl.col("high"),
    origin=train_df["Close"]
)
fig = px.line(df, y=["sum", "origin", "low", "high"])
fig.show()
# %%
