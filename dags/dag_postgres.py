from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime

def check_db_connection():
    # Especifica el nombre de la conexión que creaste en Airflow para la base de datos "testing"
    db_connection_id = 'azure_conn_id'
    db_hook = PostgresHook(postgres_conn_id=db_connection_id)
    
    try:
        conn = db_hook.get_conn()
        cursor = conn.cursor()
        # Realiza una operación sencilla, como una consulta SELECT 1
        cursor.execute("SELECT 1")
        # Obtén el resultado de la consulta (en este caso, debería ser 1)
        result = cursor.fetchone()
        cursor.close()  # Cierra el cursor
        conn.close()  # Cierra la conexión
        # Si la conexión y consulta fueron exitosas, puedes registrar un mensaje
        print("La conexión a la base de datos 'testing' fue exitosa. Resultado de la consulta: ", result)
    except Exception as e:
        # En caso de error, puedes registrar un mensaje de error
        print(f"Error al conectar a la base de datos 'testing': {str(e)}")

# Definir los argumentos del DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.today(),
    'retries': 0,
}

# Definir el DAG
dag = DAG(
    'check_db_connection',
    default_args=default_args,
    schedule_interval=None,  # Puedes ejecutarlo manualmente o según tus necesidades
)

# Definir la tarea que verifica la conexión
check_db_task = PythonOperator(
    task_id='check_db_connection',
    python_callable=check_db_connection,
    dag=dag,
)

if __name__ == "__main__":
    dag.cli()
