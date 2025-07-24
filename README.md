
Bu projede, `covtype.csv` veri seti kullanılarak **XGBoost (Extreme Gradient Boosting)** algoritmasıyla orman örtüsü türleri sınıflandırılmıştır. Boosting mantığı, XGBoost’un matematiksel yapısı, veri ön işleme süreci ve model başarımı detaylı şekilde açıklanmıştır.

---

## 📌 1. Ensemble Learning ve Boosting Nedir?

**Ensemble Learning**, birden fazla zayıf modelin birleşerek daha güçlü bir tahmin modeli oluşturmasıdır. Boosting ise bu yaklaşımın art arda hataları düzelterek ilerleyen versiyonudur.

### Ensemble Çeşitleri:
- **Bagging**: Paralel modeller (örn. Random Forest)
- **Boosting**: Sıralı modeller (örn. AdaBoost, XGBoost)
- **Stacking**: Farklı modellerin çıktıları ile ikinci bir model

---

## ⚡ 2. XGBoost Nedir?

**XGBoost**, hata fonksiyonunun (loss) gradyanına dayalı olarak çalışan, optimize edilmiş bir boosting algoritmasıdır.

### Avantajları:
- Çok hızlı ve paralel işlem destekli (C++ tabanlı)
- Overfitting’e karşı L1/L2 regularization içerir
- Eksik verilerle çalışabilir
- Büyük veri setleri için uygundur

---

## 🧠 3. XGBoost’un Matematiksel Mantığı

XGBoost, her tahmin iterasyonunda hatayı minimize etmeye çalışır. Ana fikir:

### Başlangıç:
- Tüm tahminler sabit bir değerden başlar:  
  \[
  \hat{y}^{(0)} = 	ext{average}(y)
  \]

### Her iterasyon için:
1. **Kayıp Fonksiyonu (Loss)** belirlenir:  
   Genelde logloss veya squared error.

2. **Gradyan (g)** ve **Hessian (h)** hesaplanır:  
   \[
   g_i = rac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad
   h_i = rac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
   \]

3. Yeni ağaç bu gradyanlara göre fit edilir.  
   Ağaç her node’da \( g, h \) kullanarak split gain hesaplar.

4. Yeni tahmin eklenir:
   \[
   \hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)
   \]
   - \( \eta \): learning rate  
   - \( f_t(x) \): t. iterasyonda öğrenilen model (ağaç)

5. Toplam hedef:
   \[
   	ext{Obj} = \sum_i L(y_i, \hat{y}_i^{(t)}) + \sum_t \Omega(f_t)
   \]
   - \( \Omega(f_t) \): model karmaşıklık cezası
   - 
## Matematik Kısmı Karışık Geldiyse

🧩 Adım 1: Başlangıç Tahmini Yapılır
Model ilk olarak hiçbir şey öğrenmeden tüm örnekler için aynı tahmini yapar.

Mesela sınıflandırma için başlangıç değeri genelde sabittir (örneğin logloss için 0.5 olabiliyor).

Yani model başta herkese “bence bu sınıf” der ama tahminleri pek isabetli değildir.

🎯 Adım 2: Hatalar (Loss) Hesaplanır
Her örnek için modelin ne kadar hata yaptığı hesaplanır.

Hangi örneklerde yanlış tahmin yaptıysa, o örneklere daha çok dikkat edilmesi gerekir.

Ama XGBoost klasik hata oranıyla değil, kayıp fonksiyonunun türevi ile çalışır.

📉 Adım 3: Gradyan ve Hessian Hesaplanır
Hataların eğimini (gradyanını) ve eğriliğini (hessian) hesaplar.

Bu ne demek?

Gradyan: “Tahminimi ne kadar ve hangi yönde değiştirmeliyim?”

Hessian: “Ne kadar hızlı veya yavaş değiştirmeliyim?”

🌳 Adım 4: Yeni Ağaç Bu Hatalara Göre Öğretilir
Şimdi bir karar ağacı modeli eğitilir.

Amaç: Bu ağacın, bir önceki modelin en kötü tahmin ettiği (gradyanı büyük olan) verileri daha iyi açıklaması.

Ağaç her dalında şunu sorar:

“Bu bölmeyi yaparsam gradyanı ne kadar azaltırım?”

“Split Gain” diye bir hesap yapılır: o split ne kadar faydalıysa, ağaç onu yapar.

➕ Adım 5: Yeni Tahminler Önceki Tahmine Eklenir
Öğrenilen yeni ağaç, eski tahminlerin üstüne eklenir:

𝑦
^
(
𝑡
)
=
𝑦
^
(
𝑡
−
1
)
+
𝜂
⋅
𝑓
𝑡
(
𝑥
)
y
^
​
  
(t)
 = 
y
^
​
  
(t−1)
 +η⋅f 
t
​
 (x)
Buradaki 
𝜂
η bir learning rate’tir → Ne kadar düzeltme yapacağımızı belirler.

Çok büyükse overfit olur, çok küçükse öğrenemez.

🔁 Adım 6: Bu İşlem Tekrar Edilir
Bu süreç 100 kere (ya da n_estimators kadar) tekrar eder.

Her iterasyon, bir önceki hataların üstüne yeni bir “düzeltici ağaç” ekler.

Sonunda hepsi birleşir ve güçlü bir model oluşturur.

📦 Ekstra: Model Karmaşıklığına Ceza Verilir
Çok fazla bölünme yapan, aşırı detaylı ağaçlara ceza verilir (regularization).

Böylece overfitting engellenir.

🔚 Özetle:
Tüm örneklere başlangıçta aynı tahmin yapılır.

Modelin yaptığı hatalar (gradyan) hesaplanır.

Yeni bir ağaç, bu hataları düzeltmek üzere eğitilir.

Yeni tahmin, eski tahmine eklenir.

Bu döngü onlarca kez tekrar eder.

Sonuç: Her adımda hataları düzelterek oluşmuş çok güçlü bir model.



---

## 📊 4. Kullanılan Veri Seti: `covtype.csv`

| Özellik         | Açıklama |
|------------------|----------|
| Kayıt sayısı     | 581,012  |
| Sütun sayısı     | 55       |
| Hedef değişken   | Cover_Type (1–7 arası) |
| Tür              | Çoklu sınıflı sınıflandırma |

### Sınıflar:
1. Spruce/Fir  
2. Lodgepole Pine  
3. Ponderosa Pine  
4. Cottonwood/Willow  
5. Aspen  
6. Douglas-fir  
7. Krummholz  

Veri, Colorado’daki Roosevelt Ulusal Ormanı’ndan toplanmıştır. Coğrafi ve toprak verilerine göre bitki örtüsünü tahmin etme hedeflenmiştir.

---

## 🔧 5. Veri Ön İşleme

1. **Dengeleme**: Her sınıftan eşit örnek alınarak dengeli veri seti oluşturuldu.
```python
n = df["Cover_Type"].value_counts().min()
df_balanced = df.groupby("Cover_Type", group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
```

2. **Sınıflar 0'dan başlatıldı**:
```python
y = df_balanced["Cover_Type"] - 1
```

3. **Eğitim-test ayrımı**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```

---

## 🤖 6. Model Kurulumu

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)
```

---

## 📈 7. Model Performansı

| Metrik     | Değer     |
|------------|-----------|
| Accuracy   | %85.65    |
| Precision, Recall, F1 | Tüm sınıflar için detaylı |

Confusion Matrix ve Feature Importance görsellerle desteklenmiştir.

---

## 🔬 8. Feature Importance

```python
xgb.feature_importances_
```

Görsel grafikle en önemli 15 özellik çizdirildi.  
En kritik özellik: **Elevation**. Diğerleri mesafe ve ışık sütunları.

---

## ⚔️ 9. AdaBoost – XGBoost – RandomForest Karşılaştırması

| Özellik     | AdaBoost     | XGBoost        | Random Forest |
|-------------|--------------|----------------|----------------|
| Boosting mi? | ✅          | ✅             | ❌ (Bagging)   |
| Temel Yöntem | Hatalı örnek ağırlığı | Gradyan ve 2. türev | Bootstrap örnekleme |
| Performans   | Orta         | Yüksek         | Orta-Yüksek    |
| Regularization | Yok       | Var (L1/L2)     | Yok            |

---
