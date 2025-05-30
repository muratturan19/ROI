# data_templates.py

# Özet ve ROI Şablonu
ozet_roi_data = [
    ["Özet ve ROI Analizi"],
    [],
    ["🔹 Proje Yatırım Maliyetleri"],
    [],
    ["Otomasyon Ekipmanları", 12000000],
    ["Yazılım ve Entegrasyon", 450000],
    ["Danışmanlık ve Eğitim", 35000],
    ["Altyapı Değişiklikleri", 5000],
    ["Diğer Maliyetler", 45000],
    [],
    ["Toplam Yatırım Maliyeti", "=SUM(B5:B9)"],
    [],
    ["🔹 Yıllık Getiriler"],
    [],
    ["Yıllık Maliyet Tasarrufu", 0],
    ["Yıllık Verimlilik Artışı", 0],
    ["Yıllık Kalite İyileştirme", 0],
    ["Toplam Yıllık Getiri", "=SUM(B15:B17)"],
    [],
    ["🔹 Finansal Analiz"],
    ["Toplam Yatırım Maliyeti", "=B11"],
    ["ROI (%)", "=IF(B11>0,B18/B11,0)"],
    ["Geri Ödeme Süresi (Yıl)", "=IF(B18>0,B11/B18,0)"],
    ["Net Bugünkü Değer (NPV)", "=SUM(F4:INDEX(F4:F8,B33))-B11"],
    [],
    ["🔹 Alternatif Senaryo (Banka Faizi)"],
    ["Banka Faizi ile Elde Edilecek Getiri", "=IF(B11>0,B11*(1+B32)^B33,0)"],
    ["Banka Faizi NPV", "=SUM(OFFSET(I3,1,0,B33,1))-B11"],
    [],
    [],
    ["🔹 Parametreler"],
    ["Faiz Oranı (İskonto Oranı)", 0.4],
    ["Analiz Süresi (Yıl)", 3],
    ["Yıllık İşçilik Maaş Artışı", 0.5]
]

# Maliyet Tasarrufu Şablonu
maliyet_tasarrufu_data = [
    ["Maliyet Tasarrufu Hesaplama"],
    [],
    ["🔹 Mevcut Durum"],
    [],
    ["Mevcut İşçi Sayısı", 0],
    ["Ortalama Aylık Maaş", 0],
    ["Vardiya Sayısı", 0],
    [],
    ["🔹 Otomasyon Sonrası"],
    [],
    ["Otomasyon Sonrası İşçi Sayısı", 0],
    ["Ortalama Aylık Maaş", 0],
    ["Vardiya Sayısı", 0],
    ["Yıllık Maliyet Tasarrufu", 0]
]

# Verimlilik Artışı Şablonu
verimlilik_artisi_data = [
    ["Verimlilik Artışı Hesaplama"],
    [],
    ["🔹 Mevcut Durum"],
    [],
    ["Maksimum Günlük Kapasite (adet/gün)", 0],
    ["OEE (%)", 0],
    ["Yıllık Çalışma Günü", 0],
    [],
    ["🔹 Otomasyon Sonrası"],
    [],
    ["Yeni Maksimum Kapasite (adet/gün)", 0],
    ["Yeni OEE (%)", 0],
    ["Yeni Yıllık Çalışma Günü", 0],
    [],
    ["🔹 Birim Ürün Fiyatı"],
    [],
    ["Ürün Birim Fiyatı", 0],
    [],
    ["🔹 Sonuç"],
    [],
    ["Yıllık Verimlilik Artışı", 0]
]

# Kalite İyileştirme Şablonu
kalite_iyilestirme_data = [
    ["Kalite İyileştirme Hesaplama"],
    [],
    ["🔹 Mevcut Durum"],
    [],
    ["Yıllık İade Ürün Sayısı", 0],
    ["Ortalama İade Maliyeti", 0],
    ["Yıllık Müşteri Şikayet Sayısı", 0],
    ["Ortalama Şikayet Maliyeti", 0],
    [],
    ["🔹 Otomasyon Sonrası"],
    [],
    ["Yıllık İade Ürün Sayısı", 0],
    ["Ortalama İade Maliyeti", 0],
    ["Yıllık Müşteri Şikayet Sayısı", 0],
    ["Ortalama Şikayet Maliyeti", 0],
    [],
    ["🔹 Sonuç"],
    [],
    ["Yıllık Kalite İyileştirme", "=(B5*B6+B7*B8)-(B12*B13+B14*B15)"]
]

# NPV ve ROI Bilgi Şablonu
npv_roi_bilgi_data = [
    ["NPV ve ROI Bilgi"],
    [],
    ["🔹 NPV (Net Bugünkü Değer) Nedir?"],
    ["Net Bugünkü Değer (NPV), bir yatırımın bugünkü değerini hesaplamak için kullanılan bir yöntemdir. Gelecekteki nakit akımlarını iskonto oranı ile bugüne indirger ve yatırım maliyetini çıkarır."],
    [],
    ["Formül:"],
    ["NPV = Σ (Nakit Akımı / (1 + İskonto Oranı)^Yıl) - İlk Yatırım Maliyeti"],
    [],
    ["🔹 ROI (Yatırım Getirisi) Nedir?"],
    ["ROI, bir yatırımın getirisini yatırım maliyetine oranlayarak hesaplanan bir performans ölçütüdür."],
    [],
    ["Formül:"],
    ["ROI (%) = (Toplam Yıllık Getiri / Toplam Yatırım Maliyeti) * 100"],
    [],
    ["🔹 Geri Ödeme Süresi Nedir?"],
    ["Geri Ödeme Süresi, yatırımın maliyetini karşılaması için gereken süreyi ifade eder."],
    [],
    ["Formül:"],
    ["Geri Ödeme Süresi (Yıl) = Toplam Yatırım Maliyeti / Toplam Yıllık Getiri"]
]
