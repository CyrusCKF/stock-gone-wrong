import csv
import io
from ftplib import FTP
from pathlib import Path
from typing import Iterable, cast

import pandas as pd


def get_us_tickers():
    """Retrieve all tickers listed on Nasdaq, NYSE and AMEX"""
    # list of tickers: https://quant.stackexchange.com/questions/1640/where-to-download-list-of-all-common-stocks-traded-on-nyse-nasdaq-and-amex/1862#1862
    ftp_host = "ftp.nasdaqtrader.com"
    nasdaq_file = Path("Symboldirectory/nasdaqlisted.txt")
    other_file = Path("Symboldirectory/otherlisted.txt")

    tickers: list[str] = []
    with FTP(ftp_host) as ftp:
        ftp.login()

        nasdaq_io = io.BytesIO()
        ftp.retrbinary(f"RETR {nasdaq_file}", nasdaq_io.write)
        nasdaq_io.seek(0)
        nasdaq_reader = csv.DictReader(io.TextIOWrapper(nasdaq_io), delimiter="|")
        for row in nasdaq_reader:
            tickers.append(row["Symbol"])

        other_io = io.BytesIO()
        ftp.retrbinary(f"RETR {other_file}", other_io.write)
        other_io.seek(0)
        other_reader = csv.DictReader(io.TextIOWrapper(other_io), delimiter="|")
        for row in other_reader:
            tickers.append(row["ACT Symbol"])
    return tickers


def process_history(
    df: pd.DataFrame, keep_cols: Iterable[str] = ("Close", "Volume"), add_flow=True
):
    """Flow is defined as close x volume

    Expect `df` to be have format like `yf.Tickers.history`
    """
    if add_flow:
        flow_df = df["Close"] * df["Volume"]
        flow_df.columns = pd.MultiIndex.from_product([["Flow"], flow_df.columns])
        df = pd.concat([df, flow_df], axis=1)

    # filter columns
    columns = list(keep_cols) + ["Flow"] if add_flow else []
    df = df[columns]
    df.columns = pd.MultiIndex.from_product(
        [columns, df.columns.levels[1]], names=df.columns.names
    )
    return df


def shorlist_history(df: pd.DataFrame, num: int, metrics="Volume"):
    mean_metrics = cast(pd.Series, df[metrics].mean(axis=0))
    top_metrics = mean_metrics.sort_values(ascending=False).head(num)
    top_tickers = top_metrics.index.tolist()
    df_filtered = df.loc[:, df.columns.get_level_values(1).isin(top_tickers)]
    return df_filtered
