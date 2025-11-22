from prometheus_client import start_http_server, Gauge
import time
import random


mlflow_accuracy = Gauge('mlflow_model_accuracy', 'Model accuracy from MLflow')
mlflow_loss = Gauge('mlflow_model_loss', 'Model loss from MLflow')

if __name__ == "__main__":
    
    start_http_server(8000)
    print("MLflow metrics exporter running on http://localhost:8000/metrics")

    
    while True:
        mlflow_accuracy.set(random.uniform(0.8, 1.0))
        mlflow_loss.set(random.uniform(0.0, 0.3))
        time.sleep(5)
