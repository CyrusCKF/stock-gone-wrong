import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

from langchain_ollama import ChatOllama
import numpy as np
import ollama
import yfinance as yf
from dearpygui import dearpygui as dpg
from langchain_community.retrievers import BM25Retriever

from stock_gone_wrong.event.agent import load_links, model_qa
from stock_gone_wrong.event.search_news import search_news
from stock_gone_wrong.event.signal import (
    find_largest_changes,
    find_stock_peaks,
    remove_saddle,
)


class Event(NamedTuple):
    start_time: datetime
    end_time: datetime
    start_value: float
    end_value: float


@dataclass
class NewsViewState:
    events: list[Event]
    ticker: str | None = None
    event_idx: int | None = None


_state = NewsViewState([])


def add_date_button(tag_prefix: str):
    btn = dpg.add_button(label="<Not selected>", tag=f"{tag_prefix}_btn")

    with dpg.popup(btn, mousebutton=dpg.mvMouseButton_Left) as p:

        def on_click(s, a, u):
            dpg.configure_item(p, show=False)
            date_str = f"{a['year']+1900}-{a['month']+1}-{a['month_day']}"
            dpg.configure_item(u, label=date_str, user_data=a)

        dpg.add_date_picker(
            tag=f"{tag_prefix}_picker",
            user_data=btn,
            callback=on_click,
            default_value={"year": 125, "month": 0, "month_day": 1},
        )
    return


def add_news_view():
    dpg.add_separator(label="Options")
    with dpg.group(horizontal=True, horizontal_spacing=40):
        with dpg.group():
            dpg.add_text("Enter your ticker")
            dpg.add_input_text(tag="news_ticker_input", width=100)

        with dpg.group():
            dpg.add_text("Start date")
            add_date_button("start_date")

        with dpg.group():
            dpg.add_text("End date")
            add_date_button("end_date")

        with dpg.group():
            dpg.add_text("Num events")
            dpg.add_input_int(
                width=100,
                default_value=5,
                min_value=1,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                tag="num_events",
            )

        dpg.add_button(
            label="Explore stock",
            height=50,
            width=130,
            user_data=[
                "news_ticker_input",
                "start_date_picker",
                "end_date_picker",
                "num_events",
            ],
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

        with dpg.group(horizontal=True, tag="event_selector", show=False):
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
                dpg.add_combo([], tag="model_options", enabled=False)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="refresh", width=75, callback=update_llm_models
                    )
                    dpg.add_button(
                        label="OK",
                        width=75,
                        user_data="model_options",
                        callback=find_news,
                    )
                dpg.add_text(
                    "Unable to get Ollama models. Do you forget to open it?",
                    tag="ollama_error",
                    wrap=220,
                    color=(255, 0, 0),
                )

        dpg.add_text("", tag="event_description")
        dpg.add_spacer(height=10)
        dpg.add_text("Event insights from news:", tag="news_header", show=False)
        dpg.add_text("", tag="news_summary", wrap=600)
        with dpg.group(horizontal=True, tag="news_loader", show=False):
            dpg.add_loading_indicator()
            dpg.add_text("This may take a few minutes ...")

    update_llm_models()


def slice_events(inputs: list):
    ticker_name = inputs[0]
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

    dpg.configure_item("event_selector", show=False)
    dpg.set_value("news_ticker_input", ticker_name)
    start_date_str = start_date.strftime("%Y-%m-%d")
    dpg.configure_item("start_date_btn", label=start_date_str)
    dpg.set_value("start_date_picker", start_date_map)
    end_date_str = end_date.strftime("%Y-%m-%d")
    dpg.configure_item("end_date_btn", label=end_date_str)
    dpg.set_value("end_date_picker", end_date_map)
    dpg.set_value("num_events", num_events)

    ticker = yf.Ticker(ticker_name)
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

    # update state
    _state.ticker = ticker_name
    _state.events = []
    for i, (s, e) in enumerate(changes):
        _state.events.append(Event(df.index[s], df.index[e], data[s], data[e]))

    # update view
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
    dpg.configure_item("event_selector", show=True)
    update_event_idx(None)


def update_event_idx(idx: int | None):
    if idx == _state.event_idx:
        return

    prev_idx = _state.event_idx
    _state.event_idx = idx
    if prev_idx is not None:
        dpg.configure_item(
            f"event_rect_{prev_idx}", fill=(100, 100, 100, 30), color=(100, 100, 100)
        )
    if idx is not None:
        dpg.configure_item(f"event_rect_{idx}", fill=(255, 0, 0, 30), color=(255, 0, 0))
        dpg.set_value("slice_idx", idx + 1)
        event = _state.events[idx]
        dpg.set_value(
            "event_description",
            f"{_state.ticker} {event.start_time.strftime('%Y-%m-%d')} {event.start_value:.2f}"
            f" ~ {event.end_time.strftime('%Y-%m-%d')} {event.end_value:.2f}",
        )
    else:
        dpg.set_value("event_description", "")


def update_llm_models():
    try:
        ollama_list = ollama.list()
        model_names = [m.model for m in ollama_list.models if m is not None]
        if len(model_names) == 0:
            raise RuntimeError("No models found.")
        dpg.configure_item("ollama_error", show=False)
        dpg.configure_item("model_options", items=model_names, enabled=True)
    except:
        dpg.configure_item("ollama_error", show=True)
        dpg.configure_item("model_options", items=[], enabled=False)


def find_news(sender, app_data, user_data):
    model_name = dpg.get_value(user_data)
    if model_name == "" or _state.event_idx is None:
        return

    async def ask_llm():
        await asyncio.sleep(3)
        print("Agent started")
        event = _state.events[_state.event_idx]
        search_results = search_news(
            f"{_state.ticker} stock price", event.start_time, event.end_time
        )

        links: list[str] = [r.url for r in search_results]
        splitted_docs = load_links(links[:5])
        if len(splitted_docs) < 4:
            dpg.configure_item("news_loader", show=False)
            dpg.set_value("news_summary", "Not enough news content for insights")
            return

        retriever = BM25Retriever.from_documents(splitted_docs)

        chat = ChatOllama(model=model_name)
        verb = "rise" if event.start_value < event.end_value else "drop"
        query = f"Why does the stock price of {_state.ticker} {verb}?"
        response, similar_docs = model_qa(chat, query, retriever)
        print("Agent results", response.content)
        dpg.configure_item("news_loader", show=False)
        dpg.set_value("news_summary", response.content)

    dpg.configure_item("model_modal", show=False)
    dpg.configure_item("news_loader", show=True)
    dpg.set_value("news_summary", "")
    dpg.configure_item("news_header", show=True)
    asyncio.run(ask_llm())


def __main():
    dpg.create_context()
    with dpg.window(width=700, height=500):
        add_news_view()

    dpg.set_value("news_ticker_input", "MSFT")
    start_date = datetime(2024, 1, 1)
    start_date_str = start_date.strftime("%Y-%m-%d")
    start_date_map = {
        "year": start_date.year - 1900,
        "month": start_date.month - 1,
        "month_day": start_date.day,
    }
    dpg.configure_item("start_date_btn", label=start_date_str)
    dpg.set_value("start_date_picker", start_date_map)
    end_date = datetime(2024, 12, 31)
    end_date_str = end_date.strftime("%Y-%m-%d")
    end_date_map = {
        "year": end_date.year - 1900,
        "month": end_date.month - 1,
        "month_day": end_date.day,
    }
    dpg.configure_item("end_date_btn", label=end_date_str)
    dpg.set_value("end_date_picker", end_date_map)

    dpg.show_style_editor()
    dpg.show_item_registry()

    dpg.create_viewport(title="Stock Gone Wrong", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    __main()
