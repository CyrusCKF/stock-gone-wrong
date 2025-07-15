from dataclasses import dataclass
from datetime import datetime

from dearpygui import dearpygui as dpg
import numpy as np
import yfinance as yf

from stock_gone_wrong.event.signal import (
    find_largest_changes,
    find_stock_peaks,
    remove_saddle,
)


@dataclass
class NewsViewState:
    event_idx: int | None = None


_state = NewsViewState()


def add_date_button(default_value=None, callback=None):
    btn = dpg.add_button(label="<Not selected>")

    with dpg.popup(btn, mousebutton=dpg.mvMouseButton_Left) as p:

        def on_click(s, a, u):
            dpg.configure_item(p, show=False)
            date_str = f"{a['year']+1900}/{a['month']+1}/{a['month_day']}"
            dpg.configure_item(u, label=date_str, user_data=a)
            if callback is not None:
                callback(s, a, u)

        d = dpg.add_date_picker(user_data=btn, callback=on_click)
    if default_value is not None:
        dpg.configure_item(dpg.last_item(), default_value=default_value)
        on_click(None, default_value, btn)
    return d


def add_news_view():
    dpg.add_separator(label="Options")
    with dpg.group(horizontal=True, horizontal_spacing=40):
        with dpg.group():
            dpg.add_text("Enter your ticker")
            ticker_input = dpg.add_input_text(width=100, default_value="MSFT")

        with dpg.group():
            dpg.add_text("Start date")
            start_date_btn = add_date_button({"month_day": 1, "year": 124, "month": 0})

        with dpg.group():
            dpg.add_text("End date")
            end_date_btn = add_date_button({"month_day": 31, "year": 124, "month": 11})

        with dpg.group():
            dpg.add_text("Num events")
            num_events_input = dpg.add_input_int(
                width=100,
                min_clamped=True,
                default_value=5,
                min_value=1,
                enabled=False,
                tag="num_events",
            )

    dpg.add_button(
        label="Explore stock",
        height=50,
        width=200,
        user_data=[ticker_input, start_date_btn, end_date_btn, num_events_input],
        callback=lambda s, a, u: slice_events(dpg.get_values(u)),
    )

    dpg.add_separator(label="Results")
    with dpg.group():
        with dpg.plot(label="Stock trend", height=300, width=-1, tag="trend_plot"):
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Day",
                scale=dpg.mvPlotScale_Time,
                auto_fit=True,
            )
            with dpg.plot_axis(dpg.mvYAxis, label="Value", auto_fit=True):
                dpg.add_candle_series([], [], [], [], [], tag="ticker_candles")

        with dpg.group(horizontal=True):
            dpg.add_text("Select event:")
            dpg.add_text("(?)", color=(222, 205, 16))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("CTRL+click to enter value.")
            dpg.add_slider_int(
                tag="slice_idx",
                min_value=1,
                max_value=1,
                clamped=True,
                callback=lambda s, a: update_event_idx(a - 1),
            )
            dpg.add_button(label="Find news")
            with dpg.popup(
                dpg.last_item(),
                modal=True,
                mousebutton=dpg.mvMouseButton_Left,
                tag="model_modal",
            ):
                dpg.add_text("Choose your Ollama model")
                dpg.add_combo(("granite3.2:8b", "llama3.2:1b"))
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="OK",
                        width=75,
                        callback=lambda: dpg.configure_item("model_modal", show=False),
                    )

        dpg.add_text("", tag="event_description")


def slice_events(inputs: list):
    ticker = inputs[0]
    start_date_map = inputs[1]
    end_date_map = inputs[2]
    num_events = inputs[3]
    start_date = datetime(
        start_date_map["year"] + 1900,
        start_date_map["month"] + 1,
        start_date_map["month_day"],
    )
    end_date = datetime(
        end_date_map["year"] + 1900,
        end_date_map["month"] + 1,
        end_date_map["month_day"],
    )

    ticker = yf.Ticker(ticker)
    df = ticker.history(period="max")
    df.index = df.index.tz_localize(None)
    df = df.loc[start_date:end_date]
    time_list = [d.timestamp() for d in df.index]

    data = df["Close"].to_numpy()
    maxima = find_stock_peaks(data, window=2)
    minima = find_stock_peaks(-data, window=2)

    extrema = np.unique(np.concat(([0, data.size - 1], minima, maxima)))
    trends = remove_saddle(data, extrema)
    changes = find_largest_changes(data, trends, num_events)

    dpg.configure_item(
        "ticker_candles",
        dates=time_list,
        opens=df["Open"].tolist(),
        closes=df["Close"].tolist(),
        lows=df["Low"].tolist(),
        highs=df["High"].tolist(),
        label=ticker,
    )

    i = 0
    while dpg.does_alias_exist(f"event_label_{i}"):
        dpg.delete_item(f"event_label_{i}")
        dpg.delete_item(f"event_rect_{i}")
        i += 1
    for i, (s, e) in enumerate(changes):
        dpg.add_plot_annotation(
            tag=f"event_label_{i}",
            label=f"{i+1}",
            default_value=(
                (time_list[s] + time_list[e]) / 2,
                (min(data[s:e]) + max(data[s:e])) / 2,
            ),
            parent="trend_plot",
        )
        dpg.draw_rectangle(
            tag=f"event_rect_{i}",
            pmin=(time_list[s], min(data[s:e])),
            pmax=(time_list[e], max(data[s:e])),
            color=(100, 100, 100),
            fill=(100, 100, 100, 30),
            parent="trend_plot",
        )

    dpg.configure_item("slice_idx", max_value=num_events)


def update_event_idx(idx: int):
    if idx == _state.event_idx:
        return

    prev_idx = _state.event_idx
    _state.event_idx = idx
    if prev_idx is not None:
        dpg.configure_item(
            f"event_rect_{prev_idx}", fill=(100, 100, 100, 30), color=(100, 100, 100)
        )
    if _state.event_idx is not None:
        dpg.configure_item(
            f"event_rect_{_state.event_idx}", fill=(255, 0, 0, 30), color=(255, 0, 0)
        )
    dpg.set_value("slice_idx", idx + 1)
    dpg.set_value("event_description", "MSFT 2025/06/01 361.80 ~ 2025/07/03 400.81")


def __main():
    dpg.create_context()
    with dpg.window(width=700, height=500):
        add_news_view()
    dpg.show_style_editor()
    dpg.show_item_registry()

    dpg.create_viewport(title="Stock Gone Wrong", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    __main()
