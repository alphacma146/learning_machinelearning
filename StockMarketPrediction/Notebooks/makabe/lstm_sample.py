# %%
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from IPython.display import display

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
# %%
data = np.array(
    [
        np.sin(i * 2 * np.pi / 75) + 0.25 * np.random.randn()
        for i in range(2000)
    ]
)
fig = px.line(data)
fig.show()
scaler = MinMaxScaler((0, 1), copy=True)
scalering_data = scaler.fit_transform(data.reshape(-1, 1))
train_data = scalering_data[:-200]
valid_data = scalering_data[-200:]
# %%


class SinDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data, self.labels = self.make_dataset(data, seq_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def make_dataset(self, data: np.array, seq_length: int) -> tuple:
        train_data = []
        label_data = []
        for t in range(data.shape[0] - seq_length - 1):
            train_data.append(data[t:seq_length + t])
            label_data.append(data[seq_length + t])

        return (
            tensor(train_data, device=DEVICE, dtype=torch.float32),
            tensor(label_data, device=DEVICE, dtype=torch.float32)
        )


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, hidden0=None):
        output, _ = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output


def set_random_seeds(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seeds(seed_value=37)
# %%
seq_size = 300
batch_size = 256
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 3
epochs = 150
learning_rate = 0.001
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
loss_vals = []
model.train()
for epoch in range(epochs):
    for input_val, label_val in DataLoader(
        SinDataset(train_data, seq_size),
        batch_size=batch_size,
        shuffle=True
    ):
        optimizer.zero_grad()
        output = model(input_val)
        loss = criterion(output, label_val)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_val:.4f}')
    loss_vals.append(loss_val)
# %%
input_data = tensor(
    train_data,
    device=DEVICE,
    dtype=torch.float32
).unsqueeze(0)
predictions = []
model.eval()
for _ in range(valid_data.shape[0]):
    input_data = input_data[-seq_size:]
    with torch.no_grad():
        output = model(input_data)
    predictions.append(output.item())
    input_data = torch.cat(
        (input_data, output.unsqueeze(0)),
        dim=1
    )
# %%
fig = px.line(loss_vals)
fig.show()
df = pl.DataFrame({
    "True_val": scaler.inverse_transform(valid_data).ravel(),
    "Pred_val": scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).ravel()
})
display(df.head(10))
fig = px.line(df, y=["True_val", "Pred_val"])
fig.show()
mse = np.sqrt(mean_squared_error(df["True_val"], df["Pred_val"]))
print(mean_squared_error.__name__, mse)
# %%
