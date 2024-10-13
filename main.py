# Sınıflandırma Modeli Değerlendirme

# Kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Veri Setinin Oluşturulması
gercek_deger = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
model_olasilik_tahmini = [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]


df = pd.DataFrame({
    'Gerçek Değer': gercek_deger,
    'Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)': model_olasilik_tahmini
})

df

# Classification Threshold değerine göre tahmin değerlerinin ayarlanması
df["Tahmin Edilen Değer"] = [1 if col >= 0.5 else 0 for col in df["Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)"]]
df = df[["Gerçek Değer", "Tahmin Edilen Değer", "Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)"]]
df


# Confusion Matrisinin Görselleştirilmesi
y = df["Gerçek Değer"]
y_pred = df["Tahmin Edilen Değer"]

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred)) # Classification Report


# Manual Hesaplama
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n",cm)

TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

# Accuary: Doğru sınıflandırma oranı => 0.8
acc = (TP + TN) / np.sum(cm)

# Precision: Pozitif sınıf (1) tahminlerinin başarı oranıdır. => 0.83
precision = TP / (TP + FP)

# Recall: Pozitif sınıfın(1) doğru tahmin edilme oranıdır. => 0.83
recall = TP / (TP + FN)

# F1 Score => 0.83
f1_score = 2 * (precision * recall) / (precision + recall)

# Alternatif Çözüm
cm.ravel()

TN_optional, FP_optional, FN_optional, TP_optional = cm.ravel()

# Accuary Alternatif => 0.8
accuary_optional = (TP_optional + TN_optional) / (TN_optional + FP_optional + FN_optional + TP_optional)

# Precision Alternatif
precision_optional = TP_optional / (TP_optional + FP_optional)

# Recall Alternatif
recall_optional = TP_optional / (TP_optional + FN_optional)

# F1 Score Alternatif
f1_score_optional = 2 * (precision_optional * recall_optional) / (precision_optional + recall_optional)

# Tüm Sürecin Fonksiyonlaştırılması
def confusion_matrix_calculate(gercek_deger, model_olasilik_tahmini, threshold=0.5):
    # DataFrame Oluşturma
    df = pd.DataFrame({
        'Gerçek Değer': gercek_deger,
        'Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)': model_olasilik_tahmini
    })

    # Tahmin edilen değerleri eşiğe göre ayarlama
    df["Tahmin Edilen Değer"] = [1 if prob >= threshold else 0 for prob in model_olasilik_tahmini]
    df = df[["Gerçek Değer", "Tahmin Edilen Değer", "Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)"]]


    y = df["Gerçek Değer"]
    y_pred = df["Tahmin Edilen Değer"]

    # Karışıklık matrisinin grafiğini çizme
    def plot_confusion_matrix(y, y_pred):
        acc = round(accuracy_score(y, y_pred), 2)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt=".0f")
        plt.xlabel('Tahmin Edilen Değer')
        plt.ylabel('Gerçek Değer')
        plt.title('Accuracy Score: {0}'.format(acc), size=10)
        plt.show()

    plot_confusion_matrix(y, y_pred)
    print("Classification Report:\n", classification_report(y, y_pred))

    # Confusion Matrisini Manuel Olarak Hesaplama
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Accuary: Doğru sınıflandırma oranı => 0.8
    acc = (TP + TN) / np.sum(cm)
    # Precision: Pozitif sınıf (1) tahminlerinin başarı oranıdır. => 0.83
    precision = TP / (TP + FP)
    # Recall: Pozitif sınıfın(1) doğru tahmin edilme oranıdır. => 0.83
    recall = TP / (TP + FN)
    # F1 Score => 0.83
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Alternatif Çözüm
    TN_optional, FP_optional, FN_optional, TP_optional = cm.ravel()

    # Accuary Alternatif => 0.8
    accuary_optional = (TP_optional + TN_optional) / (TN_optional + FP_optional + FN_optional + TP_optional)
    # Precision Alternatif
    precision_optional = TP_optional / (TP_optional + FP_optional)
    # Recall Alternatif
    recall_optional = TP_optional / (TP_optional + FN_optional)
    # F1 Score Alternatif
    f1_score_optional = 2 * (precision_optional * recall_optional) / (precision_optional + recall_optional)

    # Manuel Hesaplama Sonucları
    print("Confusion Matrix:\n", cm)
    print("#" * 50)
    print("Manuel Hesaplama:")
    print("#" * 50)
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print("#" * 50)

    # Alternatif Hesaplama Sonucları
    print("Confusion Matrix:\n", cm)
    print("#" * 50)
    print("Alternatif Hesaplama:")
    print("#" * 50)
    print(f"Accuracy: {accuary_optional:.2f}")
    print(f"Precision: {precision_optional:.2f}")
    print(f"Recall: {recall_optional:.2f}")
    print(f"F1 Score: {f1_score_optional:.2f}")


# Example usage
gercek_deger = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
model_olasilik_tahmini = [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]

confusion_matrix_calculate(gercek_deger, model_olasilik_tahmini, threshold=0.5)
