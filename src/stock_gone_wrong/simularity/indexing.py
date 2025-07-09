import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def create_index(vectors: np.ndarray, m=10):
    """Using IndexIVFPQ to achieve lower memory size and faster qeury"""
    dim = vectors.shape[-1]
    assert dim % m == 0
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, 100, m, 8)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 10
    return index


def create_index_meta(
    is_series_used: list[bool], tickers: list[str], dates: list[datetime]
):
    """Mapping from index to its ticker and start date"""
    meta_data = []
    ticker_span = len(is_series_used) // len(tickers)
    for i, used in enumerate(is_series_used):
        if not used:
            continue
        ticker_idx = i // ticker_span
        date_idx = i % ticker_span
        meta_data.append([tickers[ticker_idx], dates[date_idx].date()])
    meta_df = pd.DataFrame(meta_data, columns=["Ticker", "Start"])
    return meta_df


@dataclass
class DataPack:
    """Support saving and loading stock data as indices.
    Can also map search results index to its data."""

    history: pd.DataFrame
    indices: dict[str, faiss.Index]
    meta: pd.DataFrame

    def archive(self, file: Path | str):
        """Create a zip file to store all contents"""
        for name, index in self.indices.items():
            faiss.write_index(index, f"{name}.index")
        self.history.to_csv("history.csv")
        self.meta.to_csv("meta.csv")

        files = ["history.csv", "meta.csv"] + [f"{n}.index" for n in self.indices]
        with zipfile.ZipFile(file, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path)
                Path(path).unlink()

    @staticmethod
    def extract(file: Path | str) -> "DataPack":
        with zipfile.ZipFile(file, "r") as zf:
            with zf.open("history.csv") as hf:
                history_df = pd.read_csv(
                    hf, header=[0, 1], index_col=0, parse_dates=True
                )
            with zf.open("meta.csv") as mf:
                meta_df = pd.read_csv(mf, parse_dates=True, date_format="%Y/%m/%d")

            indices: dict[str, faiss.Index] = {}
            for f in zf.namelist():
                if f in ("history.csv", "meta.csv"):
                    continue
                with zf.open(f) as cf:
                    reader = faiss.PyCallbackIOReader(cf.read)
                    indices[f.removesuffix(".index")] = faiss.read_index(reader)
        return DataPack(history_df, indices, meta_df)

    def get_series(self, idx: int, metrics: str, size: int):
        """Get the data series indicated by search results `i`"""
        info = self.meta.loc[idx]
        start_pos = self.history.index.get_loc(info["Start"])
        series = (
            self.history[(metrics, info["Ticker"])]
            .iloc[start_pos : start_pos + size]
            .to_numpy()
        )
        return series
