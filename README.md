💡 Proje Hakkında
Otomasyon ROI Hesaplama Aracı, endüstriyel dönüşüm projelerinde yatırımın geri dönüşünü (ROI) hızlı ve kullanıcı dostu bir şekilde hesaplamak üzere geliştirilmiş bir Excel rapor üreticisidir.

Bu araç, hem teknik karar vericilerin hem de saha yöneticilerinin, otomasyon yatırımlarının maliyet tasarrufu, verimlilik artışı ve alternatif senaryo karşılaştırmaları ile stratejik kararlar almasına yardımcı olur.

🚀 Temel Özellikler
📊 ROI, Geri Ödeme Süresi, NPV ve diğer temel metrikleri otomatik hesaplar

💱 Para birimi seçimi desteği (₺, €, $)

🧮 İş gücü, vardiya, maaş değişkenlerine dayalı yıllık kazanç projeksiyonu

📉 Alternatif banka faizi senaryosu ile karşılaştırmalı analiz

🧾 Gelişmiş Excel çıktısı: Tablolar, grafikler, renkli biçimlendirme, dinamik formüller

🧠 Kod yapısı modüler: calculations.py, formatting.py, xlsx_report.py, gui.py

🖼️ Ekran Görüntüsü
<p align="center"> <img src="docs/screenshot.png" alt="Otomasyon ROI Arayüzü" width="600"/> </p>
🔧 Kurulum
bash
Kopyala
Düzenle
git clone https://github.com/muratturan19/ROI.git
cd ROI
pip install -r requirements.txt
python main.py
xlsxwriter modülü sisteminizde yüklü değilse:

bash
Kopyala
Düzenle
pip install xlsxwriter
📁 Dosya Yapısı
Dosya	Açıklama
main.py	Uygulama başlatıcısı
gui.py	Qt GUI tanımı, kullanıcı girişi
calculations.py	Finansal formüller
xlsx_report.py	Excel çıktısı ve rapor motoru
formatting.py	Hücre stilleri, renkler, grafik biçimi
data_templates.py	Tabloların şablon yapısı

🎯 Örnek Kullanım Senaryosu
Şirket bilgilerini ve otomasyon sonrası parametreleri girin.

Para birimi seçin.

“ROI Hesapla” butonuna tıklayın.

Anında Excel raporu oluşturulsun.

🤖 Yapay Zekâ Desteğiyle
Bu proje, gelişmiş bir Codex destekli yazılım üretim süreciyle geliştirildi.
Kod önerileri, refactor işlemleri ve test süreçleri AI destekli olarak yürütüldü.
Ayrıca proje yöneticisinin özel asistanı Mira, süreç boyunca rehberlik sağladı.

👤 Geliştirici
Murat Turan
GitHub
Delta Proje | Endüstriyel Otomasyon Direktörü

📜 Lisans
Bu proje açık kaynak değildir. Kullanım ve yayma hakkı geliştirici iznine tabidir.
Kurumsal kullanım veya dağıtım için iletişime geçin.
