import time
from pyspark.sql import SparkSession
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Kendi şema dosyanı import et
try:
    from schema import get_schema
except ImportError:
    print("HATA: schema.py dosyası bulunamadı.")
    exit()

# 1. Metrikleri Göndereceğimiz Pushgateway'i Ayarla
registry = CollectorRegistry()
pushgateway_address = 'localhost:9091'
job_name = 'telekom_overlap_job' # Grafana'da göreceğin işin adı

# 2. Göndereceğimiz Metrikleri Tanımla
# Örnek: İşin ne kadar sürdüğünü tutacak bir 'Gauge' metriği
g_job_duration = Gauge('spark_job_duration_seconds', 
                       'Spark işinin toplam çalışma süresi (saniye)', 
                       registry=registry)

# Örnek: İşlenen toplam satır sayısını tutacak bir metrik
g_total_rows = Gauge('spark_job_total_rows', 
                     'İşlenen toplam satır sayısı', 
                     registry=registry)


print("Spark Session başlatılıyor...")
start_time = time.time() # İşin başlangıç zamanını kaydet

spark = SparkSession.builder.appName("PrometheusDeneme").getOrCreate()

try:
    # 3. Spark İşini Yap (Senin verilerinle)
    
    # Overlap_matrix.csv dosyasını oku
    # (Bu dosyanın 'deneme_spark.py' ile aynı klasörde olduğundan emin ol)
    df_overlap = spark.read.csv(
        'Overlap_matrix.csv',
        header=True,
        inferSchema=True # Küçük dosya olduğu için şemayı kendi çıkarsın
    )
    
    # Sadece bir sayım al (basit bir Spark işi)
    total_rows = df_overlap.count()
    print(f"Overlap_matrix dosyasında {total_rows} satır bulundu.")
    
    # Metriğin değerini ayarla
    g_total_rows.set(total_rows)

    # (Burada normalde büyük CSV dosyanı okur, schema.py'yi kullanır
    #  ve df_overlap ile join (birleştirme) yapardın)
    
    print("Spark işi tamamlandı.")

finally:
    # 4. İş Bittiğinde Metrikleri Pushgateway'e Gönder
    
    # İşin ne kadar sürdüğünü hesapla ve metriği ayarla
    duration = time.time() - start_time
    g_job_duration.set(duration)
    
    print(f"İş {duration:.2f} saniye sürdü. Metrikler gönderiliyor...")
    
    # Toplanan tüm metrikleri ('duration' ve 'total_rows')
    # localhost:9091'deki Pushgateway'e 'telekom_overlap_job' adıyla gönder
    push_to_gateway(pushgateway_address, job=job_name, registry=registry)
    
    print("Metrikler başarıyla gönderildi.")
    
    # Spark'ı durdur
    spark.stop()
