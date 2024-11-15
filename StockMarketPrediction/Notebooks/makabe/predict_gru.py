# %%
# standard lib
from pathlib import Path
# third party
import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


class EncoderGRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(EncoderGRU, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_dim,
            hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, input_dim)

    def forward(self, src):
        output, hidden = self.gru(src)
        prediction = self.fc_out(output)[:, -1, :]
        return prediction, hidden


class DecoderGRU(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super(DecoderGRU, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            output_dim,
            hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        prediction = self.fc_out(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=1):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_dim = trg.shape[2]
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(DEVICE)
        _, hidden = self.encoder(src)
        input = src[:, -1, :].unsqueeze(1)
        for t in range(trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output.squeeze(1)
            input = (
                output
                if torch.rand(1).item() > teacher_forcing_ratio
                else trg[:, t, :].unsqueeze(1)
            )

        return outputs


class TimeScaleDataset(Dataset):
    def __init__(self, data, seq_length, mode):
        self.data, self.labels = self.make_dataset(
            data, seq_length, mode
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def make_dataset(
        self,
        data: np.ndarray,
        seq_length: int,
        mode: str
    ) -> tuple:
        train = []
        label = []
        match mode:
            case "sequence":
                for t in range(data.shape[0] - 2 * seq_length):
                    train.append(data[t:seq_length + t, :])
                    label.append(
                        data[seq_length + t:2 * seq_length + t, :]
                    )
            case "next":
                for t in range(data.shape[0] - seq_length):
                    train.append(data[t:seq_length + t, :])
                    label.append(data[seq_length + t, :])
        train = np.array(train).astype(np.float32)
        label = np.array(label).astype(np.float32)

        return torch.from_numpy(train), torch.from_numpy(label)


def set_random_seeds(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seeds(seed_value=37)
# %%
low_use_col = [
    "Low",  # target
    # "High",
    # "Volume",
    # "Low_change_rate",  # calc
    # "High_change_rate",  # calc
    # "Diff_20_75",  # calc
    # "Diff_75_120",  # calc
    # "Diff_20_120",  # calc
    # "Month",  # calc
    # "Day",  # calc
    # "Weekday"  # calc
]
seq_size = 100
batch_size = 64
input_size = len(low_use_col)
hidden_size = 200
num_layers = 10
epochs = 100
learning_rate = 0.001
dropout = 0.3
train_data = (
    train_df.filter(pl.col("Date").dt.year() < 2016)
    .select(low_use_col)
    .to_numpy()
)
valid_data = (
    train_df.filter(pl.col("Date").dt.year() >= 2016)
    .select(low_use_col)
    .to_numpy()
)
""" data = pl.DataFrame(np.array(
    [
        np.sin(i * 2 * np.pi / 75) + 0.25 * np.random.randn()
        for i in range(2400)
    ]
)).to_numpy()
train_data = data[:-400, :]
valid_data = data[-400:, :] """
scaler = StandardScaler(copy=True)
scaler.fit(train_df.select(low_use_col).to_numpy())
# scaler.fit(data)
train_data = scaler.transform(train_data)
valid_data = scaler.transform(valid_data)
# %%
encoder = EncoderGRU(input_size, hidden_size, num_layers, dropout)
decoder = DecoderGRU(input_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder).to(DEVICE)
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


def model_train(model, optimizer, criterion, batch_data, epochs, model_type):
    total_loss = []
    model.train()
    for epoch in range(epochs):
        # model.rnn.flatten_parameters()
        loss_val = []
        data_size = len(batch_data)
        for i, (input_val, label_val) in enumerate(batch_data):
            input_val = input_val.to(DEVICE)
            label_val = label_val.to(DEVICE)
            optimizer.zero_grad()
            match model_type:
                case "encoder":
                    output, _ = model(input_val)
                case "seq2seq":
                    output = model(
                        input_val[:, :-1, :], label_val,
                        (data_size - i) / data_size
                    )
            loss = criterion(output, label_val)
            loss.backward()
            optimizer.step()
            loss_val.append(loss.item())
        ave_loss = np.average(loss_val)
        print(f'Epoch [{epoch + 1:3d}/{epochs}], Loss: {ave_loss:.4f}')
        total_loss.append(ave_loss)

    return total_loss


batch_data = DataLoader(
    TimeScaleDataset(train_data, seq_size, "next"),
    batch_size=batch_size,
    shuffle=True
)
low_enc_loss = model_train(
    encoder,
    enc_optimizer,
    criterion,
    batch_data,
    epochs * 2,
    "encoder"
)
batch_data = DataLoader(
    TimeScaleDataset(train_data, seq_size, "sequence"),
    batch_size=batch_size,
    shuffle=True
)
low_loss = model_train(
    model,
    dec_optimizer,
    criterion,
    batch_data,
    epochs,
    "seq2seq"
)
fig = px.line(low_enc_loss)
fig.show()
fig = px.line(low_loss)
fig.show()
# %%


def evaluate(model, dataloader, criterion):
    model.eval()
    input_data = []
    predictions = []
    true_values = []
    total_loss = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = criterion(output, trg)
            total_loss += loss.item()
            input_data.append(src.cpu().numpy())
            predictions.append(output.cpu().numpy())
            true_values.append(trg.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    return (
        np.concatenate(input_data, axis=0),
        np.concatenate(predictions, axis=0),
        np.concatenate(true_values, axis=0),
        avg_loss
    )


valid_batch_data = DataLoader(
    TimeScaleDataset(valid_data, seq_size, "sequence"),
    batch_size=batch_size,
    shuffle=True
)
seq_input, seq_pred, seq_label, avg_loss = evaluate(
    model, valid_batch_data, criterion
)
# %%
num = np.random.randint(0, seq_input.shape[0])
print(f'Validation Loss: {avg_loss}')
input_val = (
    scaler
    .inverse_transform(seq_input[num, :, :])[:, :1]
    .flatten()
)
pred_val = (
    scaler
    .inverse_transform(seq_pred[num, :, :])[:, :1]
    .flatten()
)
label_val = (
    scaler
    .inverse_transform(seq_label[num, :, :])[:, :1]
    .flatten()
)
df = pl.concat(
    [
        pl.DataFrame({
            "Pred_val": input_val,
            "label_val": input_val,
            "flag": [0] * len(pred_val)
        }),
        pl.DataFrame({
            "Pred_val": pred_val,
            "label_val": label_val,
            "flag": [1] * len(pred_val)
        })
    ]
)
print(f"data_num :{num}")
fig = px.line(df.to_pandas(), y=["Pred_val", "label_val"], color="flag")
fig.show()
# %%
test_df = pl.read_csv(test_path)
train_data = (train_df.select(low_use_col).to_numpy())
scaler = StandardScaler(copy=True)
train_data = scaler.fit_transform(train_data)[-221:, :]
train_data = train_data.astype(np.float32)
test_data = test_df.to_numpy()[:, 1].reshape([-1, 1])
test_data = np.ones_like(test_data).astype(np.float32)

model.eval()
src = torch.from_numpy(train_data).unsqueeze(0).to(DEVICE)
trg = torch.from_numpy(test_data).unsqueeze(0).to(DEVICE)
output = model(src, trg, teacher_forcing_ratio=0.0)
input_val = (
    scaler
    .inverse_transform(train_data)
    .flatten()
)
output_val = (
    scaler
    .inverse_transform(output.squeeze(0).cpu().detach().numpy())
    .flatten()
)
df = pl.concat(
    [
        pl.DataFrame({
            "value": input_val,
            "flag": [0] * len(input_val)
        }),
        pl.DataFrame({
            "value": output_val,
            "flag": [1] * len(output_val)
        })
    ]
)
fig = px.line(df.to_pandas(), y="value", color="flag")
fig.show()
result_df = test_df.with_columns(
    Close=output_val
)
result_df.write_csv(r"..\..\Submit\submit_seq2seq_makabe.csv")
# %%
