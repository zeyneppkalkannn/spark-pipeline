import time
import os
import sys
from pyspark.sql import SparkSession
# PENCERELEME, ML ve FONKSİYONLAR için gerekli kütüphaneler
from pyspark.sql.functions import col, avg, sum, window, desc
from pyspark.ml.feature import VectorAssembler
# MLflow ve Prometheus kütüphaneleri
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. Ayarlar ---
BUYUK_VERI_SETI_YOLU = "tubitak_dataset_v2.csv" 
OVERLAP_DOSYASI = "Overlap_matrix.csv"

# Pipeline Parametreleri
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ML Hazirlik Pipeline (Windowing & Feature Selection)")

# --- 3. Prometheus Pushgateway Ayarları (Grafana & Data Drift için) ---
PUSHGATEWAY_ADDRESS = 'localhost:9091'
PROMETHEUS_JOB_NAME = 'ml_pipeline_monitoring'
registry = CollectorRegistry()

# Data Drift'i izlemek için Grafana'da göreceğimiz metrikler
g_job_duration = Gauge('spark_ml_job_duration_seconds', 'ML Pipeline İşlem Süresi', registry=registry)
g_job_success = Gauge('spark_ml_job_success', 'ML Pipeline Başarı Durumu (1=Başarılı, 0=Başarısız)', registry=registry)
g_avg_traffic = Gauge('spark_ml_avg_traffic_8h', 'Tüm Pencerelerdeki Ortalama 8 Saatlik Trafik (Data Drift)', registry=registry)
g_avg_drop_rate = Gauge('spark_ml_avg_erab_drop_rate_8h', 'Tüm Pencerelerdeki Ortalama 8 Saatlik Düşme Oranı (Data Drift)', registry=registry)


# --- 4. Spark İşini Başlat ---
print("Spark Session başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("MLPipeline_Windowing").getOrCreate()

# MLflow'a "yeni bir çalıştırma (run) başlatıyorum" diyoruz
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 5. PARAMETRELERİ KAYDET (MLflow) ---
        print("Parametreler MLflow'a kaydediliyor...")
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        mlflow.log_param("overlap_dataset", OVERLAP_DOSYASI)
        mlflow.log_param("overlap_esigi_yuzde", OVERLAP_ESIGI)
        mlflow.log_param("pencere_suresi", PENCERE_SURESI)

        # --- 6. VERİLERİ YÜKLE ---
        print(f"Ön işlenmiş veri yükleniyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(
            BUYUK_VERI_SETI_YOLU,
            header=True,
            inferSchema=True, # "Ön işlenmiş" veriye güvendiğimiz için şemayı tahmin et
            timestampFormat="yyyy-MM-dd HH:mm:ss" # Gerekirse bu formatı değiştir
        )
        # --- YENİ HATA AYIKLAMA KODU ---
        print("--- df_main (tubitak_dataset_v2.csv) Şeması Yazdırılıyor ---")
        df_main.printSchema()
        print("--- Şema Yazdırma Bitti ---")
        # ----------------------------------
        print(f"Overlap verisi yükleniyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True).drop("Unnamed: 0")

        # --- 7. TRANSFORMASYON 1: Overlap Eşiğini Uygula ---
        print(f"Transformasyon: Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        # --- YENİ SATIR: df_main'deki sütun adını düzelt ---
        # EĞER GERÇEK SÜTUN ADI 'Cell' İSE:
        df_main = df_main.withColumnRenamed("SectorID", "CELL") 
        # EĞER GERÇEK SÜTUN ADI BAŞKA BİR ŞEYSE, "Cell" yerine onu yazın.
        # Emin değilseniz, önce sadece df_main.printSchema() komutunu çalıştırıp sütun adlarına bakın.
        # --------------------------------------------------
        
        # --- YENİ SATIR: Traffic Volume Gbyte sütun adını da düzelt ---
        df_main = df_main.withColumnRenamed("Traffic Volume Gbyte", "Traffic_Volume_Gbyte")
        # -------------------------------------------------------------
        
        # --- 8. TRANSFORMASYON 2: Verileri Birleştir (Join) ---
        print("Ana veri ile filtrelenmiş Overlap verisi 'CELL' üzerinden birleştiriliyor...")
        df_joined = df_main.join(
            df_overlap_filtered.select("CELL", "N_CELL"), # Sadece komşu hücre bilgisiyle ilgileniyoruz
            on="CELL",
            how="left" # Ana verideki tüm satırları koru
        )

        # --- 9. TRANSFORMASYON 3: Pencereleme (Windowing) ---
        print(f"Transformasyon: {PENCERE_SURESI}'lik pencereler halinde veriler gruplanıyor...")
        
        # 8 saatlik pencereler ve HÜCRE+KOMŞU HÜCRE bazında grupla
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL") # İstediğiniz gibi "Hücrenin bulunduğu... ve overlap için"
        ).agg(
            # "Continuous Feature"ları (Sürekli Öznitelik) toplulaştır
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_8h"),
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h")
            # İhtiyaca göre schema.py'den daha fazla metrik eklenebilir
        )
        
        df_windowed = df_windowed.na.fill(0) # ML modelleri 'null' sevmez

        # --- 10. TRANSFORMASYON 4: ML'e Hazırlık (Feature Selection / VectorAssembler) ---
        print("Transformasyon: Öznitelikler 'VectorAssembler' ile ML inputu haline getiriliyor...")
        
        feature_columns = ["avg_traffic_8h", "avg_erab_drop_rate_8h", "avg_rrc_users_8h"]
        
        # VectorAssembler, bu sütunları tek bir "features" sütununda birleştirir
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        
        df_ml_ready = assembler.transform(df_windowed)
        
        # Sadece ML için gerekli sütunları seçelim
        df_final = df_ml_ready.select("window", "CELL", "N_CELL", "features")
        
        print("ML'e hazır son veri (sadece ilk 5 satır):")
        df_final.show(5, truncate=False)

        # --- 11. DATA DRIFT METRİKLERİNİ HESAPLA ---
        # "Data drift üzerinde ne kadar kaydı eğitime gerek var mı" sorusu için
        print("Data Drift/Grafana metrikleri hesaplanıyor...")
        df_drift_metrics = df_windowed.agg(
            avg("avg_traffic_8h").alias("drift_avg_traffic"),
            avg("avg_erab_drop_rate_8h").alias("drift_avg_drop_rate")
        ).first()

        drift_traffic = df_drift_metrics["drift_avg_traffic"]
        drift_drop_rate = df_drift_metrics["drift_avg_drop_rate"]

        # --- 12. METRİKLERİ KAYDET (MLflow & Prometheus) ---
        
        # 12a: MLflow'a Kaydet (Geçmiş Deney Kaydı için)
        print("Metrikler MLflow'a kaydediliyor...")
        mlflow.log_metric("drift_avg_traffic", drift_traffic)
        mlflow.log_metric("drift_avg_drop_rate", drift_drop_rate)
        mlflow.log_metric("output_row_count", df_final.count())
        
        # 12b: Prometheus'a Gönder (Canlı Grafana Paneli için)
        print("Metrikler Grafana'da izlenmek üzere Prometheus'a gönderiliyor...")
        g_avg_traffic.set(drift_traffic)
        g_avg_drop_rate.set(drift_drop_rate)
        g_job_success.set(1) # İş bu noktaya geldiyse başarılıdır

        # --- 13. ARTIFACT'LERİ KAYDET (MLflow/MinIO) ---
        ARTIFACT_PATH = "ml_ready_data_parquet"
        print(f"ML'e hazır veri '{ARTIFACT_PATH}' olarak MinIO'ya yüklenmek üzere kaydediliyor...")
        df_final.write.mode("overwrite").parquet(ARTIFACT_PATH)
        mlflow.log_artifact(ARTIFACT_PATH)

        mlflow.log_param("is_basarili", "Evet")
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı.")

    except Exception as e:
        # Hata olursa yakala ve MLflow/Prometheus'a bildir
        print(f"!!!! SPARK İŞİNDE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        g_job_success.set(0) # Başarısız
        print(f"MLflow Run ID: {run_id} hata ile tamamlandı.")
        sys.exit(1)

    finally:
        # --- 14. Pipeline'ı Bitir ---
        duration = time.time() - start_time
        print(f"İş {duration:.2f} saniye sürdü.")
        
        # Süreyi hem MLflow'a hem Prometheus'a gönder
        mlflow.log_metric("job_duration_seconds", duration)
        g_job_duration.set(duration)
        
        try:
            # Toplanan tüm Prometheus metriklerini Pushgateway'e gönder
            push_to_gateway(PUSHGATEWAY_ADDRESS, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("Prometheus metrikleri başarıyla gönderildi.")
        except Exception as e:
            print(f"Prometheus Pushgateway'e gönderilemedi (Pushgateway açık mı?): {e}")

        print("Spark Session durduruluyor.")
        spark.stop()
