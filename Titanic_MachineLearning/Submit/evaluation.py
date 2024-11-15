# standard lib
import sys
from pathlib import Path
from datetime import datetime
# third party
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

TARGET = "Survived"


def main(root: Path, correct: pd.DataFrame, result_list: list[pd.DataFrame]):
    """main

    Args:
        root (Path): 実行ルート
        correct (pd.DataFrame): 正解データパス
        result_list (list[pd.DataFrame]): csvのパス
    """
    def sort_rule(x: tuple[str, float]):
        it = x[1]
        if isinstance(it, (float, int)):
            return it
        else:
            return 0

    res_tex = (
        "Titanic - Machine Learning from Disaster\naccuracy_score\n"
        + f"{datetime.now()}\n\n"
    )
    corr_df = pd.read_csv(correct)
    res = {}

    for file in result_list:
        try:
            score = calc(corr_df[TARGET], file)
        except BaseException as e:
            score = f"{e}"

        res[file.stem] = score

    clac_list = [
        f"{file_name}".ljust(23, " ") + f"{score}"
        for file_name, score in sorted(
            res.items(),
            key=sort_rule,
            reverse=True
        )
    ]
    res_tex += "\n".join(clac_list)

    with open(root / "result.txt", "w") as f:
        f.write(res_tex)


def calc(true_value: pd.Series, file_path: Path) -> np.float64:
    """結果ファイルのDataFrame化と計算

    Args:
        true_value (pd.Series): 正解
        file_path (Path): csvファイルパス

    Returns:
        numpy.float64: 計算結果
    """
    res_df = pd.read_csv(file_path)
    pred = res_df[TARGET]
    score = accuracy_score(true_value, pred)

    return round(score, 4)


def get_exe_directory() -> Path:
    """exeファイルのパス

    Returns
    ----------
    Path
        Windows path
    """
    if getattr(sys, "frozen", False):
        ret = Path(sys.argv[0])
    else:
        ret = Path(__file__)

    return ret.parent.absolute()


if __name__ == "__main__":
    root = get_exe_directory()
    corr_data = root / Path(r"correct_label.csv")
    res_list = [
        file for file in root.glob("*.csv")
        if file.name != corr_data.name
    ]
    main(root, corr_data, res_list)
