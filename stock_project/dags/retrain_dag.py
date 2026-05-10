from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from train_all_model import train_all_model  # hàm đã viết trước đó

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_and_restart_spark",
    default_args=default_args,
    description="Retrain toàn bộ model rồi restart Spark Streaming",
    schedule_interval="0 0 * * 0",  # chạy mỗi Chủ Nhật 00:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "spark"],
) as dag:

    # ─────────────────────────────────────────
    # TASK 1: Retrain toàn bộ 9 symbol x 4 model
    # ─────────────────────────────────────────
    task_train = PythonOperator(
        task_id="train_all_models",
        python_callable=train_all_model,
    )

    # ─────────────────────────────────────────
    # TASK 2: Dừng Spark Streaming job cũ
    # ─────────────────────────────────────────
    task_stop_spark = BashOperator(
        task_id="stop_spark_streaming",
        bash_command="""
            docker exec stock-spark-master bash -c "
                PID=\$(ps aux | grep spark_streaming.py | grep -v grep | awk '{print \$2}')
                if [ -n \"\$PID\" ]; then
                    kill -9 \$PID
                    echo 'Đã dừng Spark Streaming PID: '\$PID
                else
                    echo 'Không có process nào đang chạy'
                fi
            "
        """,
    )

    # ─────────────────────────────────────────
    # TASK 3: Khởi động lại Spark Streaming
    # ─────────────────────────────────────────
    task_start_spark = BashOperator(
        task_id="start_spark_streaming",
        bash_command="""
            docker exec -d stock-spark-master spark-submit \
                --master spark://spark-master:7077 \
                --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
                /opt/bitnami/spark/stock_project/spark_jobs/spark_streaming.py
        """,
    )

    # ─────────────────────────────────────────
    # THỨ TỰ THỰC THI
    # ─────────────────────────────────────────
    task_train >> task_stop_spark >> task_start_spark