import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Veri setini yükle (şimdilik iris kullanıyoruz, sonra TÜBİTAK verinle değiştireceğiz)
data = load_iris()
X = data.data
y = data.target

# 2️⃣ Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ MLflow deneyini oluştur
mlflow.set_experiment("mlops_pipeline_demo")

# 4️⃣ İlk run (normal veri)
with mlflow.start_run(run_name="Original Data"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    print("Original data accuracy:", acc)

# 5️⃣ Veri değişimi simülasyonu (örneğin %30'unu çıkarıyoruz)
X_train_changed = X_train[:int(len(X_train)*0.7)]
y_train_changed = y_train[:int(len(y_train)*0.7)]

# 6️⃣ İkinci run (değişmiş veri)
with mlflow.start_run(run_name="Changed Data"):
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(X_train_changed, y_train_changed)
    preds2 = model2.predict(X_test)
    acc2 = accuracy_score(y_test, preds2)
    mlflow.log_metric("accuracy", acc2)
    print("Changed data accuracy:", acc2)

print("Tüm işlemler tamamlandı! Şimdi MLflow UI'yi aç.")
