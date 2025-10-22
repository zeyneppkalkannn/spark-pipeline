import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, desc
import mlflow

# --- Ayarlar ---
# Elimizde olan ve kullanacağımız veri seti
INPUT_CSV = "Overlap_matrix.csv"

# Dönüştürülmüş veriyi (sonucu) kaydedeceğimiz yer
OUTPUT_ARTIFACT_NAME = "top_10_avg_overlap_cells.csv"

# --- MLflow & MinIO Sunucu Ayarları ---
# Bu ayarlar, sunucuları başlattığımız ayarlarla aynı olmalı
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

# MLflow sunucumuzun adresi
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow arayüzünde görünecek deney adı
mlflow.set_experiment("Telekom Overlap Analizi")

# --- Spark İşini Başlat ---
print(f"Spark Session başlatılıyor. MLflow deneyi: 'Telekom Overlap Analizi'")
start_time = time.time()
spark = SparkSession.builder.appName("OverlapPipelineV1_MLflow").getOrCreate()

# MLflow'a "yeni bir çalıştırma (run) başlatıyorum" diyoruz
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 1. PARAMETRELERİ KAYDET (log_param) ---
        print(f"Parametreler MLflow'a kaydediliyor...")
        mlflow.log_param("input_dataset", INPUT_CSV)
        mlflow.log_param("spark_version", spark.version)

        # --- 2. VERİYİ YÜKLE ---
        print(f"Veri yükleniyor: {INPUT_CSV}")
        df = spark.read.csv(
            INPUT_CSV,
            header=True,
            inferSchema=True # Basit şemayı Spark'ın çıkarmasına izin ver (Halim'in onayıyla)
        ).drop("Unnamed: 0") # Gereksiz index sütununu at

        # --- 3. TRANSFORMASYON YAP ---
        # Halim'in "farklı transformerları deneyin" isteğini yerine getiriyoruz.
        # "N_CELL" (komşu hücre) bazında gruplayıp, ortalama çakışma alanını hesapla
        # ve en yüksek 10 tanesini al.
        
        print(f"Transformasyon yapılıyor: En yüksek 10 N_CELL hesaplanıyor...")
        df_transformed = df.groupBy("N_CELL") \
                           .agg(avg("Overlap_Alan%").alias("ortalama_cakisma_yuzdesi")) \
                           .orderBy(desc("ortalama_cakisma_yuzdesi")) \
                           .limit(10)
        
        # Sonucu ekrana da basalım
        print("Transformasyon sonucu Top 10:")
        df_transformed.show()

        # --- 4. METRİKLERİ KAYDET (log_metric) ---
        # Sadece bir örnek metrik: Tüm dataset'teki ortalama çakışma
        avg_overlap_all = df.agg(avg("Overlap_Alan%")).first()[0]
        
        print(f"Metrikler MLflow'a kaydediliyor...")
        mlflow.log_metric("avg_overlap_all_dataset", avg_overlap_all)
        
        # --- 5. ARTIFACT'LERİ KAYDET (log_artifact) ---
        # En önemli adım: Çıktıyı (pipeline'ın sonucunu) kaydetmek.
        # Önce sonucu yerel diske (geçici olarak) kaydetmemiz gerekiyor.
        
        # Spark sonucu tek bir CSV dosyası olarak yazamaz, bu yüzden pandas'a çeviriyoruz
        print(f"Sonuç artifact'i ({OUTPUT_ARTIFACT_NAME}) hazırlanıyor...")
        pandas_df = df_transformed.toPandas()
        pandas_df.to_csv(OUTPUT_ARTIFACT_NAME, index=False)

        # Şimdi o CSV dosyasını MLflow'a (ve MinIO'ya) yüklüyoruz
        print(f"Artifact, MLflow'a (MinIO'ya) yükleniyor...")
        mlflow.log_artifact(OUTPUT_ARTIFACT_NAME)
        
        # Geçici CSV dosyasını temizle
        os.remove(OUTPUT_ARTIFACT_NAME)

        mlflow.log_param("is_basarili", "Evet")
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı.")

    except Exception as e:
        # Hata olursa yakala ve MLflow'a kaydet
        print(f"!!!! SPARK İŞİNDE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        print(f"MLflow Run ID: {run_id} hata ile tamamlandı.")
        sys.exit(1) # Hata koduyla çık

    finally:
        # --- 6. Pipeline'ı Bitir ---
        duration = time.time() - start_time
        print(f"İş {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)
        
        print("Spark Session durduruluyor.")
        spark.stop()
