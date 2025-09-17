# Otomasyon ROI & NPV Aracı

Streamlit tabanlı bu uygulama, otomasyon projeleri için yatırım geri dönüşünü (ROI), net bugünkü değeri (NPV) ve alternatif getiri karşılaştırmalarını aylık bazda analiz eder. Kullanıcılar işçilik, hurda/hata, zaman tasarrufu ve gelir artışı girdilerini girerek proje finansallarını hızlıca değerlendirebilir.

## Özellikler

- İşçilik tasarrufunu aylık maaş artışını hesaba katarak hesaplar.
- Hurda/hata, zaman ve gelir artışını tek ekranda birleştirir.
- ROI, NPV, payback süresi ve alternatif getiri karşılaştırmasını sunar.
- Kümülatif net nakit akışı grafiği ve ay bazlı tablo görüntüler.
- Opex, amortisman ve üç hazır senaryo (Küçük Atölye, Orta Ölçek, Kurumsal) içerir.

## Finansal Formüller

- **Aylık Maaş Artışı:** `g = (1 + i)^(1/12) - 1`
- **İşçilik Tasarrufu:** `E × C_t`, `C_t = C0 × (1 + g)^(t-1)`
- **ROI (N ay):** `(Σ Aylık Fayda_t - I) / I`
- **NPV:** `Σ CF_t / (1 + r_m)^t - I`, `r_m = (1 + r)^(1/12) - 1`
- **Payback:** `Cum_t = -I + Σ CF_k`, ilk `Cum_t ≥ 0` ayı

## Yerel Kurulum

> Aşağıdaki komutlar Windows PowerShell içindir.

```powershell
# 1) Depoyu klonlayın veya yeni klasör oluşturun
# New-Item -ItemType Directory roi-npv-app; Set-Location roi-npv-app

# 2) Sanal ortamı oluşturun
py -m venv .venv
.\.venv\Scripts\Activate

# 3) Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4) Uygulamayı başlatın
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## Kullanım

1. Sol panelden başlangıç yatırımı, dönem ve iskonto oranını girin.
2. İşçilik, hurda/hata, zaman ve gelir alanlarını doldurun.
3. İsteğe bağlı olarak Opex ve amortisman bilgilerini girin.
4. Sonuç kartları, kümülatif net kazanç grafiği ve detaylı tabloyu inceleyin.
5. Hazır senaryolardan birini seçerek farklı ölçeklerde projeleri hızlıca değerlendirin.

## Git & GitHub Akışı (PowerShell)

```powershell
# Yeni repo oluşturma
mkdir roi-npv-app
Set-Location roi-npv-app

# (Bu depodaki dosyaları klasöre kopyalayın)

git init
git add .
git commit -m "feat: initial ROI/NPV app"

# GitHub üzerinde boş bir repo oluşturduktan sonra:
git remote add origin https://github.com/<kullanici>/roi-npv-app.git
git branch -M main
git push -u origin main
```

## Render.com'a Dağıtım

### Seçenek A: Dashboard Üzerinden
1. Render hesabınızda **New → Web Service** seçin.
2. GitHub hesabınızı bağlayın ve bu depoyu seçin.
3. Build komutu: `pip install -r requirements.txt`
4. Start komutu: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Plan olarak Free (veya ihtiyacınıza göre) seçin, Auto Deploy açık kalsın.

### Seçenek B: `render.yaml` ile (IaC)

Depo kökünde bulunan `render.yaml`, Render'ın gerekli ayarları otomatik okumasını sağlar. Render üzerinde yeni servis oluştururken **Use Existing Render.yaml** seçeneğini tercih edebilirsiniz.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
