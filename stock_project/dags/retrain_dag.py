from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys

sys.path.append("/opt/airflow/stock_project")

from dags.train_all_model import train_all_model

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_and_restart_spark",
    default_args=default_args,
    description="Retrain models rồi restart Spark Streaming",
    schedule="0 0 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "spark"],
) as dag:

    # ─────────────────────────────
    # TRAIN MODEL
    # ─────────────────────────────
    task_train = PythonOperator(
        task_id="train_all_models",
        python_callable=train_all_model,
    )

    # ─────────────────────────────
    # STOP SPARK STREAMING
    # ─────────────────────────────
    task_stop_spark = BashOperator(
        task_id="stop_spark_streaming",
        bash_command="""
        chmod 666 /var/run/docker.sock

        docker exec stock-spark-master bash -c '
            PID=$(ps aux | grep spark_streaming.py | grep -v grep | awk "{print $2}")
            if [ -n "$PID" ]; then
                kill -9 $PID
                echo "Stopped Spark Streaming PID=$PID"
            else
                echo "No Spark Streaming process found, skipping"
            fi
        '
        """,
    )

    # ─────────────────────────────
    # START SPARK STREAMING
    # ─────────────────────────────
    task_start_spark = BashOperator(
        task_id="start_spark_streaming",
        bash_command="""
        chmod 666 /var/run/docker.sock

        docker exec -d stock-spark-master bash -c '
            spark-submit \
                --master spark://spark-master:7077 \
                --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
                /opt/bitnami/spark/stock_project/spark_jobs/spark_streaming.py \
            >> /opt/bitnami/spark/stock_project/spark_jobs/spark_streaming.log 2>&1
        '
        echo "Spark Streaming started"
        """,
    )

    # ─────────────────────────────
    # FLOW
    # ─────────────────────────────
    task_train >> task_stop_spark >> task_start_spark