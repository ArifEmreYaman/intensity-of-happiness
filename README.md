
# Intensity of Happiness: Facial Expression Analysis & Clustering 🧠😊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-K--Means-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📖 Proje Hakkında (About)
Bu proje, insan yüzündeki mikro ifadeleri analiz ederek duygusal yoğunluğu (özellikle mutluluk ve gülümseme seviyesini) ölçmeyi ve sınıflandırmayı amaçlar. 

Proje üç ana aşamadan oluşur:
1.  **Veri Çıkarımı:** MediaPipe Face Mesh kullanılarak göz, kaş ve dudak bölgelerindeki kritik noktalar arasındaki Öklid mesafeleri (L2 Norm) hesaplanır.
2.  **Kümeleme (Clustering):** Elde edilen geometrik veriler, **K-Means Algoritması** kullanılarak etiketlenmemiş veriler üzerinde duygu durumlarına göre gruplandırılır.
3.  **Gerçek Zamanlı Takip:** Web kamerası veya video üzerinden anlık dudak ve yüz hareketleri analiz edilir.

## 🚀 Özellikler (Features)
* **Kapsamlı Yüz Analizi:** Dudak (iç/dış), gözler, kaşlar ve yüz silüeti dahil olmak üzere detaylı landmark takibi.
* **Makine Öğrenmesi Entegrasyonu:** `sklearn` kullanılarak yüz ifadelerinin otomatik sınıflandırılması (Clustering).
* **Veri Görselleştirme:** Kümeleme sonuçlarının `matplotlib` ile görselleştirilmesi.
* **Dataset Oluşturucu:** Ham resimlerden otomatik olarak `features.csv` veri seti oluşturma araçları.
* **Gerçek Zamanlı Takip:** Webcam üzerinden anlık veri toplama ve görselleştirme.

## 🛠️ Kurulum (Installation)

Projeyi çalıştırmak için gerekli kütüphaneleri yükleyin:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib
