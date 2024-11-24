import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import LOG_PATH

__all__ = ["visualize"]


def visualize():
    FILE_PATH = LOG_PATH / "result.csv"
    df = pd.read_csv(FILE_PATH, encoding="UTF-8")
    visualize_loss(df)
    visualize_accuracy(df)


def visualize_loss(df: pd.DataFrame):
    sns.set_theme(style="darkgrid")
    plt.figure()
    sns.lineplot(x="epoch", y="train_loss", data=df, label="train_loss")
    sns.lineplot(x="epoch", y="test_loss", data=df, label="test_loss")
    plt.title("Train and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(LOG_PATH / "loss.png")
    plt.close()


def visualize_accuracy(df: pd.DataFrame):
    sns.set_theme(style="darkgrid")
    plt.figure()
    sns.lineplot(x="epoch", y="accuracy", data=df, label="accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(LOG_PATH / "accuracy.png")
    plt.close()
