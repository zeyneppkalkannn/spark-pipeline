Bu proje, \*\*MLflow\*\* metriklerini \*\*Prometheus\*\* üzerinden toplayıp \*\*Grafana\*\* ile görselleştirir.  

Modelin doğruluk (\*\*accuracy\*\*) ve kayıp (\*\*loss\*\*) değerleri gerçek zamanlı izlenir.



\*\*Bileşenler:\*\*

\- `mlflow\_metrics\_exporter.py` → MLflow verilerini dışa aktarır  

\- `prometheus.yml` → Verileri toplar  

\- `grafana\_dashboard.json` → Grafikte gösterir  

