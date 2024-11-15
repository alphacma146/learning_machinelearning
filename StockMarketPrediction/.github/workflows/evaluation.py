# standard lib
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
# third party
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

LJUSTIFIED = 27


class BaseScoreCalc():
    """スコア算出基底クラス
    """

    def __init__(
        self,
        target_colname: str,
        greater: bool,
        header_text: str
    ) -> None:
        """Constructor

        Parameters
        ----------
        target_colname : str
            csvファイルのターゲットカラム名
        greater : bool
            結果を昇順するならTrue、降順ならFalse
        header_text : str
            結果ファイルのヘッダーにするテキスト
        """
        self.target_colname = target_colname
        self.greater = greater
        self.header_text = header_text
        self.launch_path = self.get_exe_directory()
        self.project_root = self.launch_path.parent.parent
        self.corr_data = self.launch_path / "correct.csv"

    def get_exe_directory(self) -> Path:
        """このファイルのパスを返す

        Note
        ----
        exe化するなら使う

        Returns
        -------
        Path
            実行ファイルパス
        """
        if getattr(sys, "frozen", False):
            ret = Path(sys.argv[0])
        else:
            ret = Path(__file__)

        return ret.parent.absolute()

    def calc(self, true_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.float64:
        """エラーチェック後に計算する

        Parameters
        ----------
        true_df : pd.DataFrame
            真値
        pred_df : pd.DataFrame
            予測値

        Returns
        -------
        np.float64
            スコアを丸めて返す
        """
        assert len(pred_df) == len(true_df), (
            f"submit length fault. -> length {len(pred_df)}"
        )
        assert len(pred_df.columns) == 2, (
            f"too many columns. -> {pred_df.columns}"
        )
        assert self.target_colname in pred_df.columns, (
            f"target column missing. -> {pred_df.columns}"
        )
        true_data = true_df[self.target_colname]
        pred_data = pred_df[self.target_colname]
        score = self.metric(true_data, pred_data)

        return round(score, 4)

    def metric(self, *args, **kwargs):
        """評価指標、継承してオーバーライドすること

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def get_metric_name(self):
        """評価指標名、継承してオーバーライドすること

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def main(self):
        """実行メイン
        """
        def sort_rule(x: tuple[str, float]):
            it = x[1]
            if isinstance(it, (float, int)):
                return it
            else:
                return 0

        # calc
        submit_list = [
            file for file in (self.project_root / "Submit").glob("*.csv")
            if file.name != self.corr_data.name
        ]
        corr_df = pd.read_csv(self.corr_data)
        res = {}
        for file in submit_list:
            try:
                score = self.calc(corr_df, pd.read_csv(file))
            except BaseException as e:
                score = f"{e}"
            finally:
                res[file.stem] = score

        # create result
        res_tex = (
            self.header_text + "\n" + self.get_metric_name()
            + f"\n{datetime.now(timezone(timedelta(hours=9)))}\n\n"
        )
        clac_list = [
            f"{file_name}".ljust(LJUSTIFIED, " ") + f"{score}"
            for file_name, score in sorted(
                res.items(),
                key=sort_rule,
                reverse=self.greater
            )
        ]
        res_tex += "\n".join(clac_list)

        with open(self.project_root / "Leaderboard" / "result.txt", "w") as f:
            f.write(res_tex)


class ScoreCalc(BaseScoreCalc):
    """スコア計算サブクラス

    Parameters
    ----------
    BaseScoreCalc : _type_
        基底クラス
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor
        """
        super().__init__(*args, **kwargs)

    def metric(self, *args, **kwargs) -> float:
        """評価指標

        Returns
        -------
        float
            スコア
        """
        return np.sqrt(mean_squared_error(*args, **kwargs))

    def get_metric_name(self) -> str:
        """評価指標名

        Returns
        -------
        str
            評価指標名
        """
        return "root_" + mean_squared_error.__name__


if __name__ == "__main__":
    target = "Close"
    greater = False
    header = "predict the movement of indices of S&P 500"
    SC = ScoreCalc(target, greater, header)
    SC.main()
