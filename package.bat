pyinstaller main.py ^
    --noconfirm ^
    --noconsole ^
    --name stock_gone_wrong ^
    --add-data "notebooks\us_stock_50.zip":.