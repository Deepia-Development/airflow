from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime
from datetime import timedelta
from get_data import Extract, Transform
from tensorflow.keras.models import load_model
import os
import pandas as pd
import tensorflow as tf

# Definir los argumentos del DAG
default_args = {
    'owner': 'Insert Table',
    'start_date': datetime.today(),  
    'retries': 0,
}

def insert_data_to_postgres(**kwargs):
    today = datetime.today()
    ti = kwargs['ti']
    trasformation_instance = Transform(time=250)
    X, y_scaler, df_all = trasformation_instance.preprocess_for_future()
    dag_directory = os.path.dirname(__file__)  
    model_path = os.path.join(dag_directory, 'movingave.h5')
    model = load_model(model_path)
    predictions = model.predict(X)
    predicted_output = predictions.reshape(-1, 1)
    predicted_value_original_scale = y_scaler.inverse_transform(predicted_output)
    output = predicted_value_original_scale[:30].flatten().tolist()
    date_list = [today + timedelta(minutes=i) for i in range(len(output))]
    df = pd.DataFrame({'fecha': date_list, 'prediction': output})
    
    # Insertar df en la base de datos PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id="azure_conn_id")  # Reemplaza "mynigga" 
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    # Iterar sobre las filas del DataFrame df e insertarlas en la base de datos
    for index, row in df.iterrows():
        cursor.execute("INSERT INTO predictions (fecha, predictions) VALUES (%s, %s)",
                       (row['fecha'], row['prediction']))
    
    conn.commit()
    cursor.close()
    conn.close()

with DAG(
    dag_id="table_insert",
    default_args=default_args,
    schedule_interval=timedelta(minutes=30),  # Se ejecutará cada n minutos
    catchup=False  # Ejecutar una vez
) as dag:
    task_insert_data = PythonOperator(
        task_id="insert_data",
        python_callable=insert_data_to_postgres,
        provide_context=True
    )

