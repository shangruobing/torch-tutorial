import random
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
import numpy as np


def get_device(device, cpu=False) -> torch.device:
    if cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available() and str(device):
        device = torch.device(f"cuda:{device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_now_datetime(timezone: str = "Asia/Shanghai") -> str:
    """
    get now datetime
    Returns:
        20231001_123030
    """
    return datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S")


def get_now_date(timezone: str = "Asia/Shanghai") -> str:
    """
    get now date
    Returns:
        20231001
    """
    return datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d")
