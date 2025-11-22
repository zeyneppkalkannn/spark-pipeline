import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
import mlflow

# --- 1. Ayarlar ---
# YENİ DOSYA ADI GÜNCELLENDİ:
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_data_280925__041025.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"

# Bu pipeline'ın çıktısı olan ara dosyanın adı
ARA_CIKTI_DOSYASI = "windowed_features_output" 

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - Veri Hazirlama (Windowing) - Full Metrics")

# --- 3. Spark İşini Başlat ---
print(f"PIPELINE 1 (Veri Hazırlama - Full Metrics) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("DataPrep_Windowing_Full").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 4. PARAMETRELERİ KAYDET (MLflow) ---
        print("Parametreler MLflow'a kaydediliyor...")
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        mlflow.log_param("overlap_dataset", OVERLAP_DOSYASI)
        mlflow.log_param("overlap_esigi_yuzde", OVERLAP_ESIGI)
        mlflow.log_param("pencere_suresi", PENCERE_SURESI)
        
        # --- 5. VERİLERİ YÜKLE ---
        print(f"Ön işlenmiş veri yükleniyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(
            BUYUK_VERI_SETI_YOLU,
            header=True,
            inferSchema=True,
            timestampFormat="yyyy-MM-dd HH:mm:ss"
        )
        
        print(f"Overlap verisi yükleniyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True).drop("Unnamed: 0")
        
        # --- 6. TRANSFORMASYONLAR (Hata Düzeltmeleri) ---
        
        # Hata düzeltme 1: 'SectorID' veya 'SITE' -> 'CELL'
        # Bu kısım hem eski hem yeni dosyayı destekler
        if "SectorID" in df_main.columns:
            print("Sütun Düzeltme: 'SectorID' -> 'CELL' olarak değiştiriliyor.")
            df_main = df_main.withColumnRenamed("SectorID", "CELL") 
        elif "SITE" in df_main.columns:
            print("Sütun Düzeltme: 'SITE' -> 'CELL' olarak değiştiriliyor.")
            df_main = df_main.withColumnRenamed("SITE", "CELL")
        
        # Hata düzeltme 2: 'Traffic Volume Gbyte' -> 'Traffic_Volume_Gbyte'
        if "Traffic Volume Gbyte" in df_main.columns:
            df_main = df_main.withColumnRenamed("Traffic Volume Gbyte", "Traffic_Volume_Gbyte")
        
        print(f"Transformasyon: Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        print("Ana veri ile filtrelenmiş Overlap verisi 'CELL' üzerinden birleştiriliyor...")
        df_joined = df_main.join(
            df_overlap_filtered.select("CELL", "N_CELL"),
            on="CELL",
            how="left"
        )
        
        print(f"Transformasyon: {PENCERE_SURESI}'lik pencereler halinde veriler gruplanıyor (Windowing)...")
        
        # --- AGGREGATION (SADECE DOSYADA KESİN VAR OLAN METRİKLER) ---
        # Hata veren metrikler temizlendi.
        
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            # 1. TRAFİK
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_volume_gbyte_8h"),
            sum("Traffic_Volume_Gbyte").alias("sum_traffic_volume_gbyte_8h"),
            
            # 2. ERAB - BAĞLANTI KALİTESİ
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("ERAB_ESTAB_ATT").alias("avg_erab_estab_attempts_8h"),
            avg("E-RAB_SETUP_SUCCESS_RATE").alias("avg_erab_setup_success_rate_8h"),
            
            # 3. RRC - KULLANICI SAYISI
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h"),
            avg("RRC_EstabSucc_PC").alias("avg_rrc_estab_success_rate_8h"),
            avg("NUM_OF_RRC_Att").alias("avg_rrc_attempts_8h"),
            avg("AVG_SimRRC_ConnUsr_COUNT").alias("avg_sim_rrc_conn_count_8h"),

            # 4. HANDOVER
            avg("HO_Succ_PC_In").alias("avg_ho_success_in_8h"),
            avg("INTRAFREQ_HO_SR").alias("avg_intrafreq_ho_sr_8h"),
            avg("INTERFREQ_HO_SR").alias("avg_interfreq_ho_sr_8h"),

            # 5. KAYNAK KULLANIMI (PRB)
            avg("DLPRBUtilization").alias("avg_dl_prb_utilization_8h"),
            avg("UL_PRB_Util_%").alias("avg_ul_prb_utilization_8h"),
            avg("DL_PRB_Util_%_HWI").alias("avg_dl_prb_util_hwi_8h"),

            # 6. SES TRAFİĞİ (VOLTE)
            avg("VOLTE_TRAFFIC_ERL").alias("avg_volte_traffic_erl_8h"),
            
            # 7. GÜRÜLTÜ / INTERFERANS (Noktalı İsimler)
            avg(col("`L.UL.Interference.Avg`")).alias("avg_ul_interference_8h"),
            avg(col("`L.UL.Interference.Max`")).alias("max_ul_interference_8h"),
            avg(col("`L.UL.Interference.Min`")).alias("min_ul_interference_8h"),
            
            # 8. SİNYAL GÜCÜ (RSSI)
            avg("AVGUL_RSSI_WEIGH_DBM_PUCCH").alias("avg_rssi_pucch_8h"),
            avg("AVGUL_RSSI_WEIGH_DBM_PUSCH").alias("avg_rssi_pusch_8h"),

            # 9. SİNYAL KALİTESİ (RSRP)
            avg("Avg_UL_RSRP_PUCCH").alias("avg_rsrp_pucch_8h"),
            avg("Avg_UL_RSRP_PUSCH").alias("avg_rsrp_pusch_8h"),
            
            # 10. DİĞERLERİ
            avg("Avg_CQI_HWI").alias("avg_cqi_8h"),
            avg("UL_PACKET_LOSS").alias("avg_ul_packet_loss_8h")
        )
        
        # Null değerleri 0 ile doldur
        df_windowed = df_windowed.na.fill(0)
        
        # --- 7. ARA ÇIKTIYI KAYDET ---
        print(f"Windowing sonucu ara dosya ({ARA_CIKTI_DOSYASI}) kaydediliyor...")
        
        # Veriyi Parquet formatında yerel diske kaydet
        df_windowed.write.mode("overwrite").parquet(ARA_CIKTI_DOSYASI)
        
        # MLflow'a yükle
        mlflow.log_artifact(ARA_CIKTI_DOSYASI)
        mlflow.log_param("is_basarili", "Evet")
        mlflow.log_param("total_metrics_count", len(df_windowed.columns))
        
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı (Pipeline 1 - Full Metrics).")
        
        # Sütunları listele
        print("\n=== WINDOWED ÖZNİTELİKLER (FEATURES) ===")
        feature_cols = [col for col in df_windowed.columns if col not in ["window", "CELL", "N_CELL"]]
        for i, feat in enumerate(feature_cols, 1):
            print(f"{i}. {feat}")
        print(f"\nToplam Feature Sayısı: {len(feature_cols)}")
        
    except Exception as e:
        print(f"!!!! PIPELINE 1'DE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        sys.exit(1)
        
    finally:
        duration = time.time() - start_time
        print(f"\nİş (Pipeline 1) {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)
        print("Spark Session durduruluyor (Pipeline 1).")
        spark.stop()