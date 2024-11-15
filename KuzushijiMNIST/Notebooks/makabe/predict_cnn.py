# %%
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from common.trainer import Trainer
from common.convnet import DeepConvNet
# %%
image_data = np.load(Path(r"..\..\Dataset\k49-train-imgs.npz"))['arr_0']
label_data = np.load(Path(r"..\..\Dataset\k49-train-labels.npz"))['arr_0']
image_data = image_data.reshape(
    -1,
    1,
    image_data.shape[1],
    image_data.shape[2]
)
image_data = image_data / 255
num_classes = np.max(label_data) + 1
label_data = np.eye(num_classes)[label_data].astype(int)
x_train, _, t_train, _ = train_test_split(
    image_data, label_data,
    test_size=0.9, random_state=42
)
x_train, x_valid, t_train, t_valid = train_test_split(
    x_train, t_train,
    test_size=0.2, random_state=42
)
print(x_train.shape, x_valid.shape)
# %%
network = DeepConvNet()
trainer = Trainer(
    network,
    x_train, t_train, x_valid, t_valid,
    epochs=20, mini_batch_size=100,
    optimizer='Adam', optimizer_param={'lr': 0.001},
    evaluate_sample_num_per_epoch=1000,
)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
# %%
test_data = np.load(Path(r"..\..\Dataset\k49-test-imgs.npz"))['arr_0']
test_data = test_data.reshape(
    -1,
    1,
    test_data.shape[1],
    test_data.shape[2]
)
test_data = test_data / 255
result_df = pl.read_csv(Path(r"..\..\Dataset\random_submit.csv"))
output = network.predict(test_data)
# %%


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


result = np.argmax(softmax(np.array(output)), axis=1)
result_df = result_df.with_columns(label=result)
result_df.write_csv(r"..\..\Submit\submit_重い10%tr_makabe.csv")
# %%
