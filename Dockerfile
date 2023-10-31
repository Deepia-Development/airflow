FROM apache/airflow:2.7.1-python3.9

ENV TZ=America/Mexico_City

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER ${AIRFLOW_UID}

ENV PYTHONPATH "${PYTHONPATH}:opt/airflow"

RUN python -m pip install --upgrade pip

RUN pip install onnxruntime

RUN pip install tensorflow


COPY . .

RUN pip install --user -r requirements.txt
