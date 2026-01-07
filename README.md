# final_homework

Model Dosyası: https://drive.google.com/drive/folders/1dzHnXbyNvGxYAaiJP2kUNAje36henk6J?usp=sharing
Proje 
Akciğer röntgen görüntülerinden zatürre (pneumonia) tespiti için derin öğrenme tabanlı ikili sınıflandırma modelleri geliştirilmesini kapsayan proje, bebekler ve yaşlılar gibi risk gruplarında erken teşhis için radyolog iş yükünü azaltmayı hedefler.

​
Veri Setleri
Ana veri seti Kaggle Chest X-Ray Pneumonia'dır: 1.583 normal ve 4.273 pneumonia görüntüsü içeren pediatrik veriler. Ek olarak CheXpert (20.241 görüntü) ve NIH (61.792 görüntü) external setlerle genelleme testi yapılmıştır. Veri dengesizliği için ağırlıklı örnekleme ve augmentasyon teknikleri uygulanmıştır.

​
Yöntem ve Deney Tasarımı
Veri %75 train, %15 validation, %10 test olarak stratified split ile ayrılmıştır. EfficientNet-B1 ve MobileNetV3-Large modelleri ImageNet pretrained olarak kullanılmış; giriş boyutları 256/512, augmentasyon seviyeleri (none/medium/strong) grid search ile test edilmiştir (toplam 12 senaryo). Eğitimde AdamW optimizer, CrossEntropy loss (label smoothing), CosineAnnealing scheduler ve early stopping uygulanmıştır.

​
Sonuçlar
Kaggle test setinde en iyi performans EfficientNet-B1 (512x512, strong aug): %98 balanced accuracy. External setlerde domain shift nedeniyle düşüş: CheXpert %60-67, NIH %66 balanced accuracy. Grad-CAM ile modelin akciğer bölgelerine odaklandığı doğrulanmıştır.

​
Kurulum ve Kullanım

    Python 3.8+, PyTorch, torchvision gereklidir.

    Veri setini Kaggle'dan indirin: https://www.kaggle.com/paultimothymooney/chest-x-ray-pneumonia.

    Kodları çalıştırın: train.py --model efficientnet_b1 --size 512 --aug strong.

    En iyi model: efficientnetb1_sz512_augstrong (val balanced acc ~0.97). Gelecek çalışmalar için multi-source training önerilir.
