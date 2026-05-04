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

for index in indices:
    print(f"Đang tải {index['symbol']}...")
    success = False