# standard lib
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
# third party
import pandas as pd
import numpy as np

TARGET = "rent"


def main(
    output_path: Path,
    correct: pd.DataFrame,
    result_list: list[pd.DataFrame]
):
    """main

    Args:
        output_path (Path): 実行ルート
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
        "Rent regression forecast for rental properties " +
        "in Tokyo 23 special wards.\nSMAPE\n" +
        f"\n{datetime.now(timezone(timedelta(hours=9)))}\n\n"
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
            reverse=False
        )
    ]
    res_tex += "\n".join(clac_list)

    with open(output_path / "result.txt", "w") as f:
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
    assert len(pred) == len(true_value), "submit length fault"
    score = np.average(np.abs(pred - true_value) / ((pred + true_value) / 2))

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
    file_root = get_exe_directory()
    project_root = file_root.parent.parent
    submit_path = project_root / "Submit"
    corr_data = file_root / Path(r"correct.csv")
    res_list = [
        file for file in submit_path.glob("*.csv")
        if file.name != corr_data.name
    ]
    main(project_root / "Leaderboard", corr_data, res_list)
