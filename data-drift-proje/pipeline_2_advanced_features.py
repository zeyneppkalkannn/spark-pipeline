import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, PCA
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. Ayarlar ---
ARA_GIRDI_DOSYASI = "windowed_features_output" 
SON_CIKTI_DOSYASI = "ml_ready_data_pca_parquet" # Çıktı adını değiştirdik

# ML için kullanılacak öznitelik sütunları
FEATURE_COLUMNS = ["avg_traffic_8h", "avg_erab_drop_rate_8h", "avg_rrc_users_8h"]
# PCA'in kaç özniteliğe indireceği
PCA_K_DEGERI = 2 

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow'da ÜÇÜNCÜ bir Deney (Experiment) adı veriyoruz
mlflow.set_experiment("Pipeline 2 - Advanced Features (VT & PCA)")

# --- 3. Prometheus Pushgateway Ayarları (Grafana) ---
PUSHGATEWAY_ADDRESS = 'localhost:9091'
PROMETHEUS_JOB_NAME = 'ml_advanced_fs_monitoring' # Grafana için yeni bir iş adı
registry = CollectorRegistry()
g_job_duration = Gauge('spark_adv_fs_job_duration_seconds', 'Advanced FS İşlem Süresi', registry=registry)
g_job_success = Gauge('spark_adv_fs_job_success', 'Advanced FS Başarı Durumu', registry=registry)
g_avg_traffic = Gauge('spark_adv_fs_avg_traffic_8h', 'Adv FS Ortalama 8 Saatlik Trafik (Data Drift)', registry=registry)

# --- 4. Spark İşini Başlat ---
print(f"PIPELINE 2 (Advanced Feature Selection) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("AdvancedFeatureSelection").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    mlflow.log_param("parent_run_id_pipeline_1", "c0681b0df69b4b6cbb5c01b836c419e4")


    try:
        # --- 5. ARA VERİYİ YÜKLE ---
        print(f"Pipeline 1'den gelen ara dosya okunuyor: {ARA_GIRDI_DOSYASI}")
        try:
            df_windowed = spark.read.parquet(ARA_GIRDI_DOSYASI)
        except Exception as e:
            print(f"HATA: '{ARA_GIRDI_DOSYASI}' okunamadı. Önce pipeline_1_data_prep.py'yi çalıştırdınız mı?")
            raise e

        # --- 6. DATA DRIFT METRİKLERİNİ HESAPLA (Değişiklik yok) ---
        print("Data Drift/Grafana metrikleri hesaplanıyor...")
        df_drift_metrics = df_windowed.agg(avg("avg_traffic_8h").alias("drift_avg_traffic")).first()
        drift_traffic = df_drift_metrics["drift_avg_traffic"] if df_drift_metrics["drift_avg_traffic"] else 0

        # --- 7. YENİ TRANSFORMASYONLAR (VectorAssembler -> VT -> PCA) ---

        # Adım 7a: Önce tüm öznitelikleri bir "geçici" vektörde topla
        print("Adım 7a: VectorAssembler çalışıyor...")
        assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="temp_features")
        df_assembled = assembler.transform(df_windowed)

        # Adım 7b: VarianceThresholdSelector'ı uygula
        # Varyansı 0.0 olan (yani herkesin değeri aynı olan) anlamsız sütunları atar
        print("Adım 7b: VarianceThresholdSelector çalışıyor...")
        # DÜZELTME: 'threshold' parametresini ayır:
        vt = VarianceThresholdSelector(featuresCol="temp_features", outputCol="vt_features")
        vt.setVarianceThreshold(0.0)
        # ------------------------------------
        vt_model = vt.fit(df_assembled)
        df_variance_filtered = vt_model.transform(df_assembled)

        # Hangi özniteliklerin seçildiğini MLflow'a kaydet
        selected_features_indices = vt_model.selectedFeatures
        selected_features = [FEATURE_COLUMNS[i] for i in selected_features_indices]
        print(f"Varyans Eşiği sonrası seçilen öznitelikler: {selected_features}")
        mlflow.log_param("vt_selected_features", selected_features)

        # Adım 7c: PCA'yı uygula (FeatureAgglomeration'ın Spark alternatifi)
        # Kalan öznitelikleri 'k' adet (biz 2 dedik) ana bileşene indirger
        print(f"Adım 7c: PCA (k={PCA_K_DEGERI}) çalışıyor...")
        pca = PCA(k=PCA_K_DEGERI, inputCol="vt_features", outputCol="features") # 'features' bizim son çıktımız
        pca_model = pca.fit(df_variance_filtered)
        df_final = pca_model.transform(df_variance_filtered).select("window", "CELL", "N_CELL", "features")

        print("ML'e hazır son (PCA uygulanmış) veri (sadece ilk 5 satır):")
        df_final.show(5, truncate=False)

        # PCA'in ne kadar başarılı olduğunu (açıklanan varyans) MLflow'a kaydet
        explained_variance = pca_model.explainedVariance.toArray().tolist()
        print(f"PCA Açıklanan Varyans: {explained_variance}")
        mlflow.log_param("pca_explained_variance", explained_variance)
        mlflow.log_param("pca_k_degeri", PCA_K_DEGERI)

        # --- 8. METRİKLERİ KAYDET (MLflow & Prometheus) ---
        print("Metrikler MLflow'a ve Prometheus'a gönderiliyor...")
        mlflow.log_metric("drift_avg_traffic", drift_traffic)
        mlflow.log_metric("output_row_count", df_final.count())
        g_avg_traffic.set(drift_traffic)
        g_job_success.set(1)

        # --- 9. SON ARTIFACT'İ KAYDET (MLflow/MinIO) ---
        print(f"ML'e hazır son veri ({SON_CIKTI_DOSYASI}) MinIO'ya yüklenmek üzere kaydediliyor...")
        df_final.write.mode("overwrite").parquet(SON_CIKTI_DOSYASI)
        mlflow.log_artifact(SON_CIKTI_DOSYASI)

        mlflow.log_param("is_basarili", "Evet")
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı (Pipeline 2 - Advanced).")

    except Exception as e:
        print(f"!!!! PIPELINE 2'DE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        g_job_success.set(0) 
        sys.exit(1)

    finally:
        duration = time.time() - start_time
        print(f"İş (Pipeline 2 - Advanced) {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)

        try:
            push_to_gateway(PUSHGATEWAY_ADDRESS, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("Prometheus metrikleri başarıyla gönderildi.")
        except Exception as e:
            print(f"Prometheus Pushgateway'e gönderilemedi (Pushgateway açık mı?): {e}")

        print("Spark Session durduruluyor (Pipeline 2 - Advanced).")
        spark.stop()
