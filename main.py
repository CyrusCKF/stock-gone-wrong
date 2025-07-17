from dearpygui import dearpygui as dpg

from stock_gone_wrong.view.main_view import add_main_view


def __main():
    dpg.create_context()
    with dpg.window(tag="main_window"):
        add_main_view()

    dpg.create_viewport(title="Stock Gone Wrong", width=800, height=500)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    __main()
