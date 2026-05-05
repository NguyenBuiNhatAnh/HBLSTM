from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
import time

tv = TvDatafeed()

indices = [
    {'symbol': 'DJI',     'exchange': 'TVC'},
    {'symbol': 'SPX',     'exchange': 'TVC'},
    {'symbol': 'NDX',     'exchange': 'NASDAQ'},
    {'symbol': 'DXY',     'exchange': 'TVC'},
    {'symbol': 'NI225',   'exchange': 'TVC'},
    {'symbol': '000001',  'exchange': 'SSE'},
    {'symbol': 'NIFTY',   'exchange': 'NSE'},
    {'symbol': 'UKX',     'exchange': 'TVC'},
    {'symbol': 'DEU40',   'exchange': 'TVC'},
]

os.makedirs('stock_data', exist_ok=True)

for index in indices:
    print(f"Đang tải {index['symbol']}...")
    success = False

    for attempt in range(3):
        try:
            data = tv.get_hist(
                symbol=index['symbol'],
                exchange=index['exchange'],
                interval=Interval.in_15_minute,  # ← 15 phút
                n_bars=6000
            )

            if data is not None:
                filename = f"stock_data/{index['symbol']}.csv"
                data.to_csv(filename)
                print(f"✅ Đã lưu: {filename} ({len(data)} dòng)")
                success = True
                break
            else:
                print(f"⚠️  Lần {attempt+1}: không có data, thử lại sau 5s...")
                time.sleep(5)

        except Exception as e:
            print(f"❌ Lần {attempt+1} lỗi: {e}")
            time.sleep(5)

    if not success:
        print(f"⛔ Bỏ qua {index['symbol']} sau 3 lần thử")

    time.sleep(2)

print("\nHoàn tất! Kiểm tra thư mục 'stock_data'")