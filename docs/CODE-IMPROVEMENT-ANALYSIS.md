# Kod İyileştirme Analizi

**Bağlam**: Kalman + IoU takipçisinin refaktörünü inceleyerek takip doğruluğunu veya kararlılığını zayıflatabilecek hesaplama risklerini, parametre boşluklarını ve yeniden düzenleme fırsatlarını ortaya çıkarmak. Bulgular önem derecesine göre sıralanmıştır.

## Yüksek Öncelikli Bulgular

1. **Soft IoU matematiği maliyet varsayımlarını bozuyor**  
   - Konum: `kalman_iou_tracker.cpp:130-132`  
   - Sorun: `calculateSoftIoU`, `[0,1]` aralığı yerine `[1, e]` aralığında değer üreten `std::exp(iou)` döndürüyor. Macar algoritması maliyeti `1 - iou` olarak hesaplıyor, dolayısıyla Soft IoU negatif maliyet üretip `> 0.9` eşik kontrolünü etkisiz bırakıyor.  
   - Öneri: Skor `[0,1]` aralığında kalacak şekilde sınırlı bir yumuşak IoU (ör. `(iou / (1 + alpha * (1 - iou)))` veya sigmoid) kullanın. Değişiklikten sonra eşleştirme mantığı testlerini yeniden çalıştırın.

2. **Kalman gürültü parametreleri uygulanmıyor**  
   - Konum: `kalman_iou_tracker.cpp:33-47`, `kalman_iou_tracker.cpp:593-612`  
   - Sorun: `createKalmanFilter()` içinde `processNoise`, `measurementNoise` ve `errorCovPost` sabit değerleniyor; `KalmanIoUConfig` dikkate alınmıyor. Çalışma zamanında yapılan konfigürasyon değişiklikleri mevcut veya yeni takipçilere yansımıyor.  
   - Öneri: Her filtre oluşturulurken `config.processNoise`, `config.measurementNoise` ve `config.errorCovPost` kullanılsın; `setConfig()` çağrısında mevcut filtreler yeniden başlatılmayı veya güncellemeyi düşünsün.

3. **Varsayılan kamera sınırları geçerli tespitleri siliyor**  
   - Konum: `kalman_iou_tracker.h:41-46`, `kalman_iou_tracker.cpp:337-349`, `impression.cpp:52-57`  
   - Sorun: `config.cameraBounds` dışına çıkan izler (varsayılan `640x480`) siliniyor; ancak kareler `640x640`’a yeniden boyutlandırılıyor. Y=480 altındaki her tespit kare içinde olmasına rağmen sınır dışı sayılıp siliniyor.  
   - Öneri: Yeniden boyutlandırmanın ardından (`frame.size()`) `cameraBounds` güncellensin ya da açık bir sınır verilmedikçe kontrol devre dışı bırakılabilsin.

4. **Trajektori yönetimi tutarsız ve draw içinde durum değiştiriyor**  
   - Konum: `kalman_iou_tracker.cpp:204-212`, `kalman_iou_tracker.cpp:285-288`, `kalman_iou_tracker.cpp:515-519`, `kalman_iou_tracker.h:54`  
   - Sorunlar:  
     - İz geçmişi kırpma işlemi `config.maxTrajectoryLength` yerine sabit `35` kullanıyor.  
     - `draw()` her çizimde trajektoriyi değiştirdiği için, trajektori açıldığında ekleme frekansı iki katına çıkıyor ve görselleştirme mantığı takip durumu ile iç içe geçiyor.  
   - Öneri: Trajektori bakımı takip güncelleme adımı içinde merkezileştirilsin, yapılandırılabilir uzunluk dikkate alınsın ve `draw()` yan etkisiz kalsın.

5. **IoU yardımcıları hâlâ exception fırlatabiliyor**  
   - Konum: `kalman_iou_tracker.cpp:16-30`, `kalman_iou_tracker.cpp:80-93`  
   - Sorun: `calculateIoU` ve `calculateGIoU`, birleşim veya kaplama alanı sıfıra düştüğünde exception atıyor. Çoğu çağrı noktası geçersiz kutuları süzse de bozulmuş Kalman durumları (ör. sıfır genişlik) uygulamayı yine de sonlandırıyor.  
   - Öneri: Exception yerine `0.0` döndürüp günlük veya sayaçla raporlayın; böylece takipçi kontrollü şekilde bozulur.

## Orta Öncelikli Bulgular

1. **Eşleştirme kapısı IoU eşiği ile tutarsız**  
   - Konum: `kalman_iou_tracker.cpp:189-200`  
   - Sorun: `cost > 0.9` olduğunda eşleşme reddediliyor; bu IoU `< 0.1` anlamına geliyor (`cost = 1 - iou`). `config.iouThreshold` değeri `0.2` olduğundan, eşiği geçen her eşleşme aynı zamanda bu kapıdan da geçiyor. Soft IoU’da maliyet negatif olduğundan kapı tamamen etkisiz.  
   - Öneri: Red maliyetini `config.iouThreshold` ile ilişkilendirin (örn. `cost > 1 - config.iouThreshold`) ve standart olmayan IoU türleri için maliyet fonksiyonunu yeniden değerlendirin.

2. **`KalmanIoUConfig::IoUType::AIOU` uygulanmamış**  
   - Konum: `kalman_iou_tracker.cpp:73-75`  
   - Sorun: AIOU seçildiğinde sessizce `0.0` döndürülüyor, tüm eşleşmeler kapatılıyor.  
   - Öneri: Alfa IoU formülünü uygulayın veya desteklenmeyen varyantların seçilmesini engellemek için konfigürasyonu doğrulayın.

3. **Sınıf bilgisi kayboluyor**  
   - Konum: `kalman_iou_tracker.cpp:414-421`, `KalmanIoUConfig`  
   - Sorun: `TrackedResult.classId`, tespitler sınıf ID’si taşımasına rağmen `-1`’e sabitleniyor; aşağı akış bileşenleri sınıfa göre filtre yapamıyor.  
   - Öneri: `TrackedObject` içinde son eşleşen tespitin sınıf ID’si (ve gerekirse güveni) saklanarak sonuçlara iletilsin.

4. **Trajektori uzunluğu konfigürasyonu kullanılmıyor**  
   - Konum: `kalman_iou_tracker.cpp:210-211`, `kalman_iou_tracker.cpp:286-287`, `kalman_iou_tracker.cpp:583-585`  
   - Sorun: `setTrajectoryLength()` yalnızca `config.maxTrajectoryLength` değerini güncelliyor; takip mantığı hâlâ `35`'te kırpıyor.  
   - Öneri: Sabitler `config.maxTrajectoryLength` ile değiştirilsin ve konfigürasyon değiştikten sonra mevcut trajektoriler sınırlandırılsın.

5. **Tahmin tekrar kullanımı legacy yolda max-lost korumasını yok sayıyor**  
   - Konum: `kalman_iou_tracker.cpp:292-301`  
   - Sorun: Macar algoritması kullanılmayan güncelleme yolu, `maxLostFrames` ne olursa olsun tahminleri kullanmaya devam ediyor; oysa Macar yolunda bu limitten sonra duruluyor.  
   - Öneri: Aynı korumayı uygulayın veya tutarsız durumları önlemek için eski yöntemi kullanımdan kaldırın.

## Düşük Öncelikli / Refaktör Fırsatları

- **Görselleştirme durumu yönetiyor** (`kalman_iou_tracker.cpp:497-553`): Görselleştirmeyi saf görünüm tutmak için çizimden önce iz verilerini kopyalamayı düşünün.
- **`Impression` içinde sahiplik** (`impression.cpp:5-27`): Erken çıkışlarda sızıntıları önlemek için manuel `new/delete` yerine `std::unique_ptr` tercih edin.
- **Kullanılmayan takip çıktısı** (`impression.cpp:86-107`, `147-169`): Takip sonuçları hesaplanıyor ancak kullanılmıyor; bunları görselleştirmeye iletmek ileride takip iç durumunun tekrar okunmasını engeller.
- **Tekrarlanan renk tohumlama** (`kalman_iou_tracker.cpp:383-385`): Her takipçi için mevcut zaman damgasıyla `cv::RNG` oluşturmak aynı millisaniyede yaratılan izlere aynı rengi üretebilir; RNG örneğini önbelleğe alın.

## Önerilen Sonraki Adımlar

1. Soft IoU implementasyonunu düzeltip tüm IoU varyantlarını kapsayan regresyon testleri ekleyin.
2. `KalmanIoUConfig` parametrelerini filtre başlatmalarına bağlayıp konfigürasyon değiştiğinde mevcut takipçileri güncelleyin.
3. Kamera sınırlarını veri odaklı hale getirin (örn. her `update()` çağrısında son kare boyutuna göre ayarlayın) ve temizleme kontrolünü devre dışı bırakmak için bir seçenek sunun.
4. Trajektori mutasyonunu `draw()`dan kaldırın; geçmiş bakımını takip döngüsü içinde tutup yapılandırılabilir sınırları gözetin.
5. Sıfır alanlı ölçümleri simüle eden birim veya entegrasyon testleri ekleyerek takipçinin artık exception atmadığını doğrulayın.
