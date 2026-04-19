CREATE TABLE IF NOT EXISTS stock_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    actual_price FLOAT,
    predicted_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index để Web truy vấn nhanh theo mã chứng khoán và thời gian
CREATE INDEX idx_symbol_time ON stock_predictions (symbol, timestamp);