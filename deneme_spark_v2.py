import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# 1. Kendi şema dosyamızı import ediyoruz
# (schema.py dosyasının bu betikle aynı klasörde olduğundan emin ol)
try:
    from schema import get_schema
except ImportError:
    print("HATA: schema.py dosyası bu dizinde bulunamadı.")
    exit()

# -------------------------------------------------------------------
# !! DEĞİŞTİR !!: Büyük veri setinin yolunu buraya yaz
# -------------------------------------------------------------------
BUYUK_VERI_SETI_YOLU = "BURAYA_BUYUK_CSV_DOSYANIZIN_YOLUNU_YAZIN.csv"
# Örn: "/Users/zeynep/Desktop/SparkProjem/buyuk_telekom_verisi.csv"
# -------------------------------------------------------------------


# --- Prometheus Ayarları ---
registry = CollectorRegistry()
pushgateway_address = 'localhost:9091'
job_name = 'telekom_pipeline_job_v2' # Grafana'da göreceğin işin adı

# --- Göndereceğimiz Yeni Metrikleri Tanımla ---
# 'Gauge', anlık bir değeri tutan metrik türüdür (örn. sıcaklık, doluluk oranı)

# 1. İşin toplam çalışma süresi
g_job_duration = Gauge('spark_job_duration_seconds', 
                       'Spark işinin toplam çalışma süresi (saniye)', 
                       registry=registry)

# 2. İşin başarılı olup olmadığı (1=başarılı, 0=başarısız)
g_job_success = Gauge('spark_job_success', 
                      'İşin başarılı olup olmadığı (1=başarılı, 0=başarısız)', 
                      registry=registry)

# 3. schema.py'den gelen veriler için YENİ metrikler
g_total_traffic = Gauge('spark_total_traffic_gbyte', 
                        'Tüm hücrelerdeki toplam Trafik Hacmi (Gbyte)', 
                        registry=registry)

g_avg_drop_rate = Gauge('spark_avg_erab_drop_pc', 
                        'Ortalama E-RAB Düşme Yüzdesi (%)', 
                        registry=registry)

g_avg_users = Gauge('spark_avg_rrc_users', 
                    'Ortalama RRC Kullanıcı Sayısı', 
                    registry=registry)


# --- Spark İşini Başlat ---
print("Spark Session başlatılıyor...")
start_time = time.time() # İşin başlangıç zamanını kaydet
spark = SparkSession.builder.appName("TelekomPipelineV2").getOrCreate()

try:
    # --- 1. Verileri Yükle ---
    
    # schema.py'den şemayı al
    main_schema = get_schema()
    
    print(f"Büyük veri seti yükleniyor: {BUYUK_VERI_SETI_YOLU}")
    # Büyük ana veri setini, schema.py'yi kullanarak yükle
    df_main = spark.read.csv(
        BUYUK_VERI_SETI_YOLU,
        schema=main_schema,
        header=True,
        # ÖNEMLİ: CSV'deki tarih formatın buysa (değilse değiştirmen gerekir):
        timestampFormat="yyyy-MM-dd HH:mm:ss" 
    )

    print("Overlap_matrix.csv yükleniyor...")
    # Overlap_matrix.csv dosyasını oku (Bu dosya da aynı klasörde olmalı)
    df_overlap = spark.read.csv(
        'Overlap_matrix.csv',
        header=True,
        inferSchema=True # Bu dosya küçük, şemayı kendi çıkarsın
    ).drop("Unnamed: 0") # Baştaki gereksiz index sütununu at

    
    # --- 2. Verileri Birleştir (Join) ---
    print("Veriler 'CELL' sütunu üzerinden birleştiriliyor...")
    df_joined = df_main.join(
        df_overlap,
        on="CELL",        # Hangi sütundan birleşecek
        how="left"       # Ana verideki tüm satırları koru
    )

    # --- 3. Hesaplamaları Yap (Aggregation) ---
    print("Hesaplamalar yapılıyor (Toplam Trafik, Ortalama Düşme Oranı)...")
    
    # Tüm veri seti üzerinden tek bir özet satırı oluştur
    df_agg = df_joined.agg(
        sum("Traffic_Volume_Gbyte").alias("total_traffic"),
        avg("ERAB_Drop_PC").alias("avg_drop_rate"),
        avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_users")
    )

    # df_agg.show() # Sonucu görmek istersen yorum satırını kaldırabilirsin

    # --- 4. Metrik Değerlerini Ayarla ---
    print("Hesaplanan metrikler toplanıyor...")
    # Hesaplanan tek satırlık sonucu al
    results = df_agg.first()

    if results:
        # Metrik 'Gauge'lerine değerlerini ata
        g_total_traffic.set(results["total_traffic"])
        g_avg_drop_rate.set(results["avg_drop_rate"])
        g_avg_users.set(results["avg_users"])
        
        print(f"Toplam Trafik: {results['total_traffic']}")
        print(f"Ort. Düşme Oranı: {results['avg_drop_rate']}")
        print(f"Ort. Kullanıcı: {results['avg_users']}")
        
        # İş başarılı oldu
        g_job_success.set(1)
    else:
        print("UYARI: Veri setinden sonuç hesaplanamadı.")
        g_job_success.set(0) # İş başarısız oldu

except Exception as e:
    # Bir hata olursa (örn. dosya bulunamadı, sütun adı yanlış)
    print(f"!!!! SPARK İŞİNDE HATA OLUŞTU !!!!")
    print(e)
    g_job_success.set(0) # İş başarısız oldu

finally:
    # --- 5. Tüm Metrikleri Pushgateway'e Gönder ---
    
    # İş ne kadar sürmüş olursa olsun süreyi hesapla
    duration = time.time() - start_time
    g_job_duration.set(duration)
    
    print(f"İş {duration:.2f} saniye sürdü. Tüm metrikler gönderiliyor...")
    
    try:
        # Toplanan tüm metrikleri (duration, success, traffic, vb.)
        # localhost:9091'deki Pushgateway'e 'telekom_pipeline_job_v2' adıyla gönder
        push_to_gateway(pushgateway_address, job=job_name, registry=registry)
        print("Metrikler başarıyla gönderildi.")
    except Exception as e:
        print(f"!!!! METRİKLER PUSHGATEWAY'E GÖNDERİLEMEDİ !!!!")
        print(f"Pushgateway'in çalıştığından emin ol (Terminal'de 'pushgateway' komutu açık mı?)")
        print(e)

    # Spark'ı durdur
    print("Spark Session durduruluyor.")
    spark.stop()
