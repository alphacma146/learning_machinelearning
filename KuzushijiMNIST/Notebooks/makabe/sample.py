# %%
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from IPython.display import display
# %%
image_data = np.load(Path(r"..\..\Dataset\k49-test-imgs.npz"))['arr_0']
label_data = np.load(
    Path(r"..\..\.github\workflows\k49-test-labels.npz"))['arr_0']
print(set(label_data))
conv_dict = {
    0: "あ", 1: "い", 2: "う", 3: "え", 4: "お",
    5: "か", 6: "き", 7: "く", 8: "け", 9: "こ",
    10: "さ", 11: "し", 12: "す", 13: "せ", 14: "そ",
    15: "た", 16: "ち", 17: "つ", 18: "て", 19: "と",
    20: "な", 21: "に", 22: "ぬ", 23: "ね", 24: "の",
    25: "は", 26: "ひ", 27: "ふ", 28: "へ", 29: "ほ",
    30: "ま", 31: "み", 32: "む", 33: "め", 34: "も",
    35: "や", 36: "ゆ", 37: "よ",
    38: "ら", 39: "り", 40: "る", 41: "れ", 42: "ろ",
    43: "わ", 44: "ゐ", 45: "ゑ", 46: "を",
    47: "ん", 48: "ゝ",
}
# %%
TARGET = 24
print(conv_dict[TARGET])
i = 0
while i < 10:
    num = np.random.randint(len(image_data))
    if label_data[num] != TARGET:
        continue
    print(num)
    sns.heatmap(image_data[num], cmap='cividis', cbar=False)
    plt.axis('off')
    plt.show()
    i += 1
# %%
image_data = np.load(Path(r"..\..\Dataset\k49-train-imgs.npz"))
print(image_data)
img_data_arr = image_data["arr_0"]
sns.heatmap(img_data_arr[0], cmap='cividis', cbar=False)
plt.axis('off')
plt.show()
# %%
correct_data = np.load(
    Path(r"..\..\.github\workflows\k49-test-labels.npz"))['arr_0']
cor_df = pd.DataFrame(correct_data)
cor_df = cor_df.reset_index()
cor_df.columns = ["index", "label"]
ram_df = cor_df.copy()
ram_df["label"] = cor_df.apply(lambda _: np.random.randint(0, 49), axis=1)
display(ram_df)
print(ram_df.shape)
ram_df.to_csv("random_submit.csv", index=False)
# %%
matrix = confusion_matrix(cor_df[["label"]], ram_df[["label"]])
print(matrix)
print(precision_score(cor_df[["label"]],
                      ram_df[["label"]],
                      average='macro',
                      zero_division=0))
print(precision_score.__name__)
# %%
