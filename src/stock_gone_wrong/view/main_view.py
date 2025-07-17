import datetime

import dearpygui.dearpygui as dpg

from stock_gone_wrong.view.news_view import add_news_view, slice_events
from stock_gone_wrong.view.pattern_view import TrendSegment, add_pattern_view


def add_main_view():
    with dpg.tab_bar(tag="tab_view"):
        with dpg.tab(tag="similarity_tab", label="Similarity"):
            add_pattern_view()
        with dpg.tab(tag="events_tab", label="Events"):
            add_news_view()
    dpg.configure_item("explore_pattern", callback=explore_pattern)


def explore_pattern(s, a, u):
    dpg.set_value("tab_view", "events_tab")
    assert isinstance(u, None | TrendSegment)
    if u is not None:
        start_date_map = {
            "year": u.start.year - 1900,
            "month": u.start.month - 1,
            "month_day": u.start.day,
        }
        end_date_map = {
            "year": u.end.year - 1900,
            "month": u.end.month - 1,
            "month_day": u.end.day,
        }
        slice_events([u.ticker, start_date_map, end_date_map, 3])


def __main():
    dpg.create_context()
    with dpg.window(width=800, height=500):
        add_main_view()

    dpg.create_viewport(title="Stock Gone Wrong", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    __main()
