from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import time
import schedule
from datetime import datetime

# Khởi tạo API (Chạy 1 lần ngoài vòng lặp để tránh bị block)
tv = TvDatafeed()

# Danh sách các mã cổ phiếu/chỉ số của bạn
indices = [
    {'symbol': 'DJI', 'exchange': 'TVC'},
    {'symbol': 'SPX', 'exchange': 'TVC'},
    # Bạn có thể thêm các mã khác vào đây...
]

def fetch_realtime_data():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] 🔄 Đang kích hoạt tiến trình lấy dữ liệu 15 phút...")
    
    for index in indices:
        try:
            # CHÚ Ý: Chạy realtime không tải 7000 dòng nữa, chỉ tải 10 dòng gần nhất 
            # để lấy được cây nến hiện tại nhanh nhất và không tốn băng thông.
            data = tv.get_hist(
                symbol=index['symbol'],
                exchange=index['exchange'],
                interval=Interval.in_15_minute,
                n_bars=10  
            )
            
            if data is not None and not data.empty:
                # Lấy dòng cuối cùng (cây nến mới nhất vừa chốt sổ)
                latest_candle = data.iloc[[-1]]
                
                # In ra màn hình để kiểm tra
                print(f"✅ {index['symbol']} - Giá mới nhất: {latest_candle['close'].values[0]}")
                
                # Ghi nối (append) dòng mới này vào file CSV lịch sử của bạn
                filename = f"stock_data/{index['symbol']}.csv"
                latest_candle.to_csv(filename, mode='a', header=False)
                
            else:
                print(f"⚠️ Không lấy được data cho {index['symbol']} lúc này.")
                
        except Exception as e:
            print(f"❌ Lỗi khi lấy {index['symbol']}: {e}")
            
        # Nghỉ 2 giây giữa các lần gọi API để tránh bị TradingView chặn IP
        time.sleep(2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Đã xong. Chờ 15 phút tiếp theo...")
    
    # Ở BƯỚC TIẾP THEO, BẠN SẼ GỌI HÀM CẬP NHẬT MÔ HÌNH HBLSTM TẠI ĐÂY
    # update_hblstm_model()

# ==========================================
# CÀI ĐẶT BỘ ĐẾM THỜI GIAN (SCHEDULER)
# ==========================================

# 1. Test thử ngay lập tức khi vừa chạy script (không cần đợi 15p đầu tiên)
fetch_realtime_data()

# 2. Cài đặt chu kỳ cứ đúng 15 phút chạy 1 lần
schedule.every(15).minutes.do(fetch_realtime_data)

# 3. Vòng lặp vô hạn để giữ chương trình luôn chạy ngầm
print("\n🚀 HỆ THỐNG STREAMING ĐÃ KHỞI ĐỘNG...")
while True:
    schedule.run_pending()
    time.sleep(1) # Nghỉ 1s để không ăn CPU