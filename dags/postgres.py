from airflow import DAG 
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime 



# Definir los argumentos del DAG
default_args = {
    'owner': 'Create Table',
    'start_date': datetime(2023, 10, 23),
    'retries': 0
}

with DAG(
  dag_id = "table_create",
  default_args = default_args
) as dag:
    task_1 = PostgresOperator(
        task_id = "Create_Table",
        postgres_conn_id = "azure_conn_id",
        sql = """ 
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            fecha VARCHAR(255),
            predictions DECIMAL(10, 2)
        );
        """
        )