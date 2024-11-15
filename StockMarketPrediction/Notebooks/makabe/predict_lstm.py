# %%
# standard lib
from pathlib import Path
from datetime import date
# third party
import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import tensor, nn
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from tqdm import tqdm
from IPython.display import display

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
# %%


def split_signal_noise(data, normal_cutoff, order=5):
    b_low, a_low = butter(order, normal_cutoff, btype='low', analog=False)
    b_high, a_high = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_low = filtfilt(b_low, a_low, data)
    filtered_high = filtfilt(b_high, a_high, data)
    return filtered_low, filtered_high


train_path = Path(r"..\..\Dataset\train_data.csv")
test_path = Path(r"..\..\Dataset\test_data.csv")
test_df = pl.read_csv(test_path)
train_df = (
    pl.read_csv(train_path)
    .with_columns(pl.col("Date").cast(pl.Date))
)
low, high = split_signal_noise(
    train_df.select([pl.col("Close")]).to_numpy().ravel(),
    0.025
)
train_df = (
    train_df
    .with_columns(
        Low=low,
        High=high,
    )
    .with_columns(
        pl.col("Date").dt.month().alias("Month"),
        pl.col("Date").dt.day().alias("Day"),
        pl.col("Date").dt.weekday().alias("Weekday"),
        pl.col("Low").rolling_mean(20).alias("20_rolling"),
        pl.col("Low").rolling_mean(75).alias("75_rolling"),
        pl.col("Low").rolling_mean(120).alias("120_rolling"),
        pl.col("Low").shift(1).alias("Low_prev_val"),
        pl.col("High").shift(1).alias("High_prev_val"),
    )
    .fill_null(0)
    .with_columns(
        Low_change_rate=(
            (pl.col("Low") - pl.col("Low_prev_val")) /
            pl.col("Low_prev_val") * 100
        ),
        High_change_rate=(
            (pl.col("High") - pl.col("High_prev_val")) /
            pl.col("High_prev_val") * 100
        ),
        Diff_20_75=pl.col("20_rolling") - pl.col("75_rolling"),
        Diff_75_120=pl.col("75_rolling") - pl.col("120_rolling"),
        Diff_20_120=pl.col("20_rolling") - pl.col("120_rolling")
    )
    .with_columns(
        Low_change_rate=(
            pl
            .when(pl.col("Low_change_rate") == float("inf"))
            .then(pl.lit(0))
            .otherwise(pl.col("Low_change_rate"))
        ),
        High_change_rate=(
            pl
            .when(pl.col("High_change_rate") == float("inf"))
            .then(pl.lit(0))
            .otherwise(pl.col("High_change_rate"))
        )
    )
)
display(train_df)
fig = px.line(train_df, x="Date", y=["Close", "Low", "High"])
fig.show()
# %%


class TimeScaleDataset(Dataset):
    def __init__(self, data, seq_length, inuput_size, output_size):
        self.data, self.labels = self.make_dataset(
            data, seq_length, inuput_size, output_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def make_dataset(
        self,
        data: np.ndarray,
        seq_length: int,
        inuput_size: int,
        output_size: int
    ) -> tuple:
        train = []
        label = []
        for t in range(data.shape[0] - seq_length - 1):
            train.append(data[t:seq_length + t, :inuput_size])
            label.append(data[seq_length + t, :output_size])
        train = np.array(train).astype(np.float32)
        label = np.array(label).astype(np.float32)

        return (
            torch.from_numpy(train).clone().to(DEVICE),
            torch.from_numpy(label).clone().to(DEVICE)
        )


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm_hidden_size = hidden_dim
        self.num_layer_size = num_layers
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, output_size)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, hidden0=None):
        b_size = inputs.size(0)  # batch size
        h0 = torch.zeros(
            self.num_layer_size,
            b_size,
            self.lstm_hidden_size
        ).to(DEVICE)
        c0 = torch.zeros(
            self.num_layer_size,
            b_size,
            self.lstm_hidden_size
        ).to(DEVICE)
        lstm_output_seq, (h_n, c_n) = self.rnn(inputs, (h0, c0))
        output = self.dropout(self.fc1(lstm_output_seq))
        output = self.fc2(output[:, -1, :])
        # output, _ = self.rnn(inputs, hidden0)
        # output = self.output_layer(output[:, -1, :])

        return output


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            output_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        _, encoder_hidden = self.encoder(src)
        decoder_output, _ = self.decoder(trg, encoder_hidden)
        output = self.output(decoder_output)

        return output


def set_random_seeds(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seeds(seed_value=37)
# %%
seq_size = 100
batch_size = 256
input_size = 4
hidden_size = 400
output_size = 1
num_layers = 5
epochs = 100
learning_rate = 0.001
dropout = 0.3
low_use_col = [
    "Low",  # target
    # "High",
    # "Volume",
    # "Low_change_rate",  # calc
    # "High_change_rate",  # calc
    "Diff_20_75",  # calc
    "Diff_75_120",  # calc
    "Diff_20_120",  # calc
    # "Month",  # calc
    # "Day",  # calc
    # "Weekday"  # calc
]
train_data = (
    train_df
    .filter(
        (pl.col("Date") >= date(2010, 6, 23))
        & (pl.col("Date") < date(2016, 1, 1))
    )
    .select(low_use_col)
    .to_numpy()
)
valid_data = (
    train_df
    .filter(pl.col("Date").dt.year() == 2016)
    .select(low_use_col)
    .to_numpy()
)
scaler_trg = StandardScaler(copy=True)
scaler_oth = StandardScaler(copy=True)
scaler_trg.fit(train_df[["Low"]].to_numpy())
scaler_oth.fit(train_df[low_use_col].to_numpy())
train_data = scaler_oth.transform(train_data)
valid_data = scaler_oth.transform(valid_data)
# %%


def model_train(model, optimizer, criterion, batch_data, epochs):
    total_loss = []
    model.train()
    for epoch in range(epochs):
        # model.rnn.flatten_parameters()
        loss_val = []
        for input_val, label_val in batch_data:
            optimizer.zero_grad()
            output = model(input_val)
            loss = criterion(output, label_val)
            loss.backward()
            optimizer.step()
            loss_val.append(loss.item())
        ave_loss = np.average(loss_val)
        print(f'Epoch [{epoch + 1:3d}/{epochs}], Loss: {ave_loss:.4f}')
        total_loss.append(ave_loss)

    return total_loss


def model_eval(model, data, valid_data, calc_func):
    model.eval()
    all_data = tensor(data, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    sequential_predictions = []
    for i in tqdm(range(valid_data.shape[0])):
        input_data = all_data[:, -seq_size:, :]
        with torch.no_grad():
            output = model(input_data)
        out_val = [output[0, i].item() for i in range(output.shape[1])]
        calc_val = calc_func(
            all_data.cpu().numpy(), valid_data, out_val)
        sequential_predictions.append(calc_val)
        calc_tensor = tensor(
            calc_val, device=DEVICE, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        all_data = torch.cat((all_data, calc_tensor), dim=1)

    all_data = tensor(data, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    valid_predictions = []
    for i in tqdm(range(valid_data.shape[0])):
        input_data = all_data[:, -seq_size:, :]
        with torch.no_grad():
            output = model(input_data)
        out_val = [output[0, i].item() for i in range(output.shape[1])]
        calc_val = calc_func(
            all_data.cpu().numpy(),
            valid_data,
            out_val
        )
        valid_predictions.append(calc_val.copy())
        calc_val[0] = valid_data[i, 0]
        calc_tensor = tensor(
            calc_val, device=DEVICE, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        all_data = torch.cat((all_data, calc_tensor), dim=1)

    return sequential_predictions, valid_predictions
# %%


def get_low_calcdata(base_data, valid_data, append_vals: list):
    df = pl.DataFrame({
        "Low": scaler_trg.inverse_transform(
            np.append(base_data[:, :, 0], append_vals[0]).reshape(-1, 1)
        ).ravel(),
        # "High": np.append(base_data[:, :, 1], append_vals[1]),
        # "Volume": np.append(base_data[:, :, 1], append_vals[1])
    }).with_columns(
        pl.col("Low").rolling_mean(20).alias("20_rolling"),
        pl.col("Low").rolling_mean(75).alias("75_rolling"),
        pl.col("Low").rolling_mean(120).alias("120_rolling"),
        pl.col("Low").shift(1).alias("Low_prev_val"),
        # pl.col("High").shift(1).alias("High_prev_val"),
    ).with_columns(
        Low_change_rate=(
            (pl.col("Low") - pl.col("Low_prev_val")) /
            pl.col("Low_prev_val")
        ),
        # High_change_rate=(
        #    (pl.col("High") - pl.col("High_prev_val")) /
        #    pl.col("High_prev_val")
        # ),
        Diff_20_75=pl.col("20_rolling") - pl.col("75_rolling"),
        Diff_75_120=pl.col("75_rolling") - pl.col("120_rolling"),
        Diff_20_120=pl.col("20_rolling") - pl.col("120_rolling")
    ).select(low_use_col)

    # return np.concatenate(
    #    [df.tail(1).to_numpy().ravel()[:1], date_val]
    # )
    return scaler_oth.transform(df.tail(1).to_numpy()).ravel()


low_model = LSTMModel(
    input_size,
    hidden_size,
    output_size,
    num_layers,
    dropout
).to(DEVICE)
low_criterion = nn.MSELoss()
low_optimizer = torch.optim.Adam(low_model.parameters(), learning_rate)
low_batch_data = DataLoader(
    TimeScaleDataset(train_data, seq_size, input_size, output_size),
    batch_size=batch_size,
    shuffle=True
)
low_loss = model_train(
    low_model,
    low_optimizer,
    low_criterion,
    low_batch_data,
    epochs
)
seq_pred, val_pred = model_eval(
    low_model, train_data, valid_data, get_low_calcdata
)
# %%
fig = px.line(low_loss)
fig.show()
t_col = 0
df = pl.DataFrame({
    "True_val": scaler_oth.inverse_transform(valid_data)[:, t_col].ravel(),
    "Pred_val": scaler_oth.inverse_transform(
        np.array(seq_pred)
    )[:, t_col].ravel()
})
display(df.head(10))
fig = px.line(df, y=["True_val", "Pred_val"])
fig.show()
mse = np.sqrt(mean_squared_error(df["True_val"], df["Pred_val"]))
print(mean_squared_error.__name__, mse)
df = pl.DataFrame({
    "True_val": scaler_oth.inverse_transform(valid_data)[:, t_col].ravel(),
    "Pred_val": scaler_oth.inverse_transform(
        np.array(val_pred)
    )[:, t_col].ravel()
})
fig = px.line(df, y=["True_val", "Pred_val"])
fig.show()
mse = np.sqrt(mean_squared_error(df["True_val"], df["Pred_val"]))
print(mean_squared_error.__name__, mse)
# %%
seq_size = 100
batch_size = 256
input_size = 2
hidden_size = 400
output_size = 1
num_layers = 3
epochs = 100
learning_rate = 0.001
dropout = 0.3
high_use_col = [
    # "Low",
    "High",  # target
    # "Volume",
    # "Low_change_rate",  # calc
    "High_change_rate",  # calc
    # "Diff_20_75",  # calc
    # "Diff_75_120",  # calc
    # "Diff_20_120",  # calc
    # "Month",  # calc
    # "Day",  # calc
    # "Weekday"  # calc
]
high_train_data = (
    train_df
    .filter(
        (pl.col("Date") >= date(2010, 6, 23))
        & (pl.col("Date") < date(2016, 1, 1))
    )
    .select(high_use_col)
    .to_numpy()
)
high_valid_data = (
    train_df
    .filter(pl.col("Date").dt.year() == 2016)
    .select(high_use_col)
    .to_numpy()
)
scaler_trg = StandardScaler(copy=True)
scaler_oth = StandardScaler(copy=True)
scaler_trg.fit(train_df[["High"]].to_numpy())
scaler_oth.fit(train_df[high_use_col].to_numpy())
high_train_data = scaler_oth.transform(high_train_data)
high_valid_data = scaler_oth.transform(high_valid_data)
# %%


def get_high_calcdata(base_data, valid_data, append_vals: list):
    df = pl.DataFrame({
        "High": scaler_trg.inverse_transform(
            np.append(base_data[:, :, 0], append_vals[0]).reshape(-1, 1)
        ).ravel(),
        # "Volume": np.append(base_data[:, :, 1], append_vals[1])
    }).with_columns(
        pl.col("High").shift(1).alias("High_prev_val"),
    ).with_columns(
        High_change_rate=(
            (pl.col("High") - pl.col("High_prev_val")) /
            pl.col("High_prev_val")
        )
    ).select(high_use_col)

    # return np.concatenate(
    #    [df.tail(1).to_numpy().ravel()[:1], date_val]
    # )
    return scaler_oth.transform(df.tail(1).to_numpy()).ravel()


high_model = LSTMModel(
    input_size,
    hidden_size,
    output_size,
    num_layers,
    dropout
).to(DEVICE)
high_criterion = nn.MSELoss()
high_optimizer = torch.optim.Adam(low_model.parameters(), learning_rate)
high_batch_data = DataLoader(
    TimeScaleDataset(high_train_data, seq_size, input_size, output_size),
    batch_size=batch_size,
    shuffle=True
)
high_loss = model_train(
    high_model,
    high_optimizer,
    high_criterion,
    high_batch_data,
    epochs
)
high_seq_pred, high_val_pred = model_eval(
    high_model, high_train_data, high_valid_data, get_high_calcdata
)
# %%
fig = px.line(high_loss)
fig.show()
t_col = 1
df = pl.DataFrame({
    "True_val": scaler_oth.inverse_transform(
        high_valid_data
    )[:, t_col].ravel(),
    "Pred_val": scaler_oth.inverse_transform(
        np.array(high_seq_pred)
    )[:, t_col].ravel()
})
display(df.head(10))
fig = px.line(df, y=["True_val", "Pred_val"])
fig.show()
mse = np.sqrt(mean_squared_error(df["True_val"], df["Pred_val"]))
print(mean_squared_error.__name__, mse)
df = pl.DataFrame({
    "True_val": scaler_oth.inverse_transform(
        high_valid_data
    )[:, t_col].ravel(),
    "Pred_val": scaler_oth.inverse_transform(
        np.array(high_val_pred)
    )[:, t_col].ravel()
})
fig = px.line(df, y=["True_val", "Pred_val"])
fig.show()
mse = np.sqrt(mean_squared_error(df["True_val"], df["Pred_val"]))
print(mean_squared_error.__name__, mse)
# %%
