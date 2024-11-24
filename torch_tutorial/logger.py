import os

import pandas as pd

from config import LOG_PATH
from parse import Arguments
from utils import get_now_datetime

__all__ = ["Logger"]


class Logger:

    @staticmethod
    def insert_row(
            args: Arguments,
            model: str,
            epoch: int,
            train_loss: float,
            test_loss: float,
            accuracy: float,
    ) -> None:
        """
        Args:
            args: argparse,
            model: Model,
            epoch: int,
            train_loss: float,
            test_loss: float,
            accuracy: float,
        """
        new_row = {
            "time": get_now_datetime(),
            "args": args.string,
            "model": model,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
        }
        FILE_PATH = LOG_PATH / "result.csv"
        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
        if os.path.exists(FILE_PATH):
            df = pd.read_csv(FILE_PATH, encoding="UTF-8")
        else:
            # print(f"The {FILE_PATH} does not exist, an {FILE_PATH} file has been created.")
            df = pd.DataFrame(columns=list(new_row.keys()))
        df.loc[len(df)] = new_row
        df.to_csv(FILE_PATH, index=False, encoding="UTF-8")
