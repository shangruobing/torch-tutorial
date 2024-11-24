import os
import sys
from os.path import dirname, abspath
from pathlib import Path

from utils import get_now_datetime

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent

DATASET_PATH = ROOT_PATH / "data"

LOG_PATH = ROOT_PATH / f"log/{get_now_datetime()}"

check_paths = [
    DATASET_PATH,
    ROOT_PATH / "log"
]

for check_path in check_paths:
    if not check_path.exists():
        os.mkdir(check_path)


def print_path():
    print("ROOT_PATH:", ROOT_PATH)
    print("DATASET_PATH:", DATASET_PATH)
    print("LOG_PATH:", LOG_PATH)
