import datetime as dt
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yfinance as yf
from dearpygui import dearpygui as dpg
from sklearn.preprocessing import minmax_scale

from stock_gone_wrong.simularity.indexing import DataPack
from stock_gone_wrong.simularity.preprocess import extended_minmax_scale
from stock_gone_wrong.simularity.ticker import process_history
from stock_gone_wrong.simularity.visual import calculate_PI

DAYS_WINDOW = 50
DATA_FILE = Path(__file__) / f"../../../../notebooks/us_stock_{DAYS_WINDOW}.zip"
if not DATA_FILE.is_file():
    DATA_FILE = Path(f"_internal/us_stock_{DAYS_WINDOW}.zip")


class TrendSegment(NamedTuple):
    ticker: str
    start: datetime
    data: list[float]

    @property
    def end(self):
        return self.start + dt.timedelta(days=DAYS_WINDOW * 2)


@dataclass
class PatternViewState:
    trends: list[TrendSegment]
    line_idx: int | None = None
    loaded_data: DataPack | None = None


_state = PatternViewState([])


def add_pattern_view():
    _create_plot_color_theme("side_theme", (125, 125, 125, 64))
    _create_plot_color_theme("side_theme_highlight", (170, 170, 170, 128))
    _create_plot_color_theme("PI_theme", (43, 173, 171, 30), dpg.mvPlotCol_Fill)
    _create_plot_color_theme("mean_theme", (43, 173, 171))
    _create_plot_color_theme("main_theme", (245, 94, 64))
    _create_plot_color_theme("last_record_theme", (64, 67, 245))

    dpg.add_separator(label="Options")
    with dpg.group(horizontal=True, horizontal_spacing=40):
        with dpg.group():
            dpg.add_text("Enter your ticker")
            dpg.add_input_text(tag="pattern_ticker_input", width=100)

        with dpg.group():
            dpg.add_text("Metrics")
            dpg.add_combo(("Close", "Volume"), tag="metrics_dropdown", width=150)

        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("Show forecast")
                dpg.add_text("(!)", color=(222, 205, 16))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(
                        "WARNING: The calculations are based on the scaled data, which will "
                        "lead to underestimating the uncertainty. The results have little "
                        "predictive power.",
                        wrap=300,
                    )
            dpg.add_checkbox(tag="forecast_toggle")

        with dpg.group():
            dpg.add_text("Num results")
            dpg.add_input_int(
                width=100,
                min_clamped=True,
                default_value=10,
                min_value=1,
                tag="num_patterns",
            )

    dpg.add_spacer(height=10)
    # all this to center the button
    with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
        dpg.add_table_column(width_stretch=True)
        dpg.add_table_column()
        dpg.add_table_column(width_stretch=True)
        with dpg.table_row():
            dpg.add_table_cell()
            dpg.add_button(
                label="Find similar stocks",
                height=50,
                width=200,
                user_data=[
                    "pattern_ticker_input",
                    "metrics_dropdown",
                    "forecast_toggle",
                    "num_patterns",
                ],
                callback=lambda s, a, u: find_similar_stocks(*dpg.get_values(u)),
            )
            dpg.add_table_cell()

    dpg.add_separator(label="Search")

    dpg.add_loading_indicator(tag="similarity_loading", show=False)

    with dpg.group(tag="similarity_plot", show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("Select trend:")
            dpg.add_text("(?)", color=(222, 205, 16))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("CTRL+click to enter value.")
            dpg.add_slider_int(
                tag="trend_idx",
                max_value=dpg.get_value("num_patterns") - 1,
                clamped=True,
                width=-1,
                callback=lambda s, a: update_line_idx(a),
            )

        with dpg.group(horizontal=True, tag="explore_group", show=False):
            dpg.add_text("", tag="side_info")
            dpg.add_button(tag="explore_pattern", user_data=None, label="Explore more")

        with dpg.plot(height=300, width=-1, crosshairs=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Day", auto_fit=True)
            with dpg.plot_axis(
                dpg.mvYAxis, label="Close", auto_fit=True, tag="plot_lines"
            ):
                dpg.add_shade_series([], [], y2=[], label="PI", tag="PI_shade")
                dpg.bind_item_theme(dpg.last_item(), "PI_theme")
                dpg.add_line_series([], [], label="Mean", tag="mean_line")
                dpg.bind_item_theme(dpg.last_item(), "mean_theme")
                dpg.add_line_series([], [], label="Main", tag="query_line")
                dpg.bind_item_theme(dpg.last_item(), "main_theme")
                dpg.add_inf_line_series([-1])
                dpg.bind_item_theme(dpg.last_item(), "last_record_theme")

            dpg.add_plot_annotation(
                label="Last record",
                default_value=(-1, 0.5),
                color=(64, 67, 245),
            )


def update_line_idx(idx: int):
    line_idx = _state.line_idx
    if idx == line_idx:
        return

    prev_idx = line_idx
    line_idx = idx
    if prev_idx is not None:
        dpg.bind_item_theme(f"side_line_{prev_idx}", "side_theme")
    if line_idx is not None:
        dpg.bind_item_theme(f"side_line_{line_idx}", "side_theme_highlight")
    dpg.set_value("trend_idx", idx)
    trend = _state.trends[idx]
    dpg.configure_item("explore_group", show=True)
    dpg.set_value("side_info", f"{trend[0]} {trend[1].strftime('%Y-%m-%d')}")
    dpg.configure_item("explore_pattern", user_data=trend)
    _state.line_idx = idx


def find_similar_stocks(
    ticker: str, metrics: str, show_forecast: bool, num_results: int
):
    print("find_similar", ticker, metrics, show_forecast, num_results)
    dpg.configure_item("similarity_loading", show=True)
    dpg.configure_item("similarity_plot", show=False)
    dpg.configure_item("trend_idx", max_value=num_results - 1)

    try:
        query_df = yf.Tickers(ticker).history("6mo", progress=False)
        if query_df is None:
            raise RuntimeError(f"Cannot find ticker data {ticker}")
    except:
        return
    query_df = process_history(query_df)
    query_df.columns = query_df.columns.droplevel(1)

    raw_query_data = query_df[metrics][:DAYS_WINDOW].to_numpy()
    query_data = minmax_scale(raw_query_data, feature_range=(0, 1))
    query_data = query_data.reshape((1, -1))

    if _state.loaded_data is None:
        _state.loaded_data = DataPack.extract(DATA_FILE)
    data_pack = _state.loaded_data
    (dist, *_), (idx, *_) = data_pack.indices[metrics].search(query_data, num_results)
    trends: list[TrendSegment] = []
    for i in idx:
        info = data_pack.meta.loc[i]
        series = data_pack.get_series(i, metrics, DAYS_WINDOW * 2)
        series = extended_minmax_scale(series, (0, 1), fit_window=slice(0, DAYS_WINDOW))
        start = datetime.strptime(info["Start"], "%Y-%m-%d")
        trends.append(TrendSegment(info["Ticker"], start, series.tolist()))
    scaled_data = np.stack([t[2] for t in trends])

    stock_days = list(range(-DAYS_WINDOW, DAYS_WINDOW))
    i = 0
    while dpg.does_alias_exist(f"side_line_{i}"):
        dpg.delete_item(f"side_line_{i}")
        i += 1
    for i, t in enumerate(trends):
        dpg.add_line_series(stock_days, t[2], tag=f"side_line_{i}", parent="plot_lines")
        dpg.bind_item_theme(dpg.last_item(), "side_theme")

    dpg.configure_item(
        "query_line", x=stock_days[:DAYS_WINDOW], y=query_data.tolist()[0]
    )

    if show_forecast:
        series_mean = scaled_data.mean(axis=0)
        pi_lower, pi_upper = calculate_PI(scaled_data)
        dpg.configure_item("mean_line", x=stock_days, y=series_mean, show=True)
        dpg.configure_item(
            "PI_shade", x=stock_days, y1=pi_lower, y2=pi_upper, show=True
        )
    else:
        dpg.configure_item("mean_line", x=[], y=[], show=False)
        dpg.configure_item("PI_shade", x=[], y1=[], y2=[], show=False)

    time.sleep(2)
    dpg.configure_item("similarity_loading", show=False)
    dpg.configure_item("similarity_plot", show=True)
    _state.trends = trends


def _create_plot_color_theme(tag, color, target=dpg.mvPlotCol_Line):
    with dpg.theme(tag=tag):
        with dpg.theme_component():
            dpg.add_theme_color(target, color, category=dpg.mvThemeCat_Plots)


def __main():
    dpg.create_context()
    with dpg.window(width=700, height=500):
        add_pattern_view()
    dpg.set_value("pattern_ticker_input", "MSFT")
    dpg.set_value("metrics_dropdown", "Close")
    dpg.set_value("forecast_toggle", True)
    dpg.show_style_editor()
    dpg.show_item_registry()

    dpg.create_viewport(title="Stock Gone Wrong", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    __main()
