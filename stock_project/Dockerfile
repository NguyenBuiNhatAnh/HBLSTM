FROM bitnamilegacy/spark:3.5.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests joblib scikit-learn minio pandas numpy matplotlib pyarrow \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# COPY toàn bộ stock_project vào trong image
COPY . /opt/bitnami/spark/stock_project/

# PYTHONPATH trỏ lên thư mục cha để import stock_project được
ENV PYTHONPATH="${PYTHONPATH}:/opt/bitnami/spark"

USER 1001