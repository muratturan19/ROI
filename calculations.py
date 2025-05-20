# calculations.py

def maliyet_tasarrufu_hesapla(maliyet_sayfasi):
    # Mevcut maliyet
    mevcut_isci = maliyet_sayfasi['B5'].value or 0
    mevcut_maas = maliyet_sayfasi['B6'].value or 0
    mevcut_vardiya = maliyet_sayfasi['B7'].value or 0
    mevcut_maliyet = mevcut_isci * mevcut_maas * mevcut_vardiya * 12  # Yıllık maliyet

    # Otomasyon sonrası maliyet
    otomasyon_sonrasi_isci = maliyet_sayfasi['B10'].value or 0
    otomasyon_sonrasi_maas = maliyet_sayfasi['B11'].value or 0
    otomasyon_sonrasi_vardiya = maliyet_sayfasi['B12'].value or 0
    otomasyon_sonrasi_maliyet = otomasyon_sonrasi_isci * otomasyon_sonrasi_maas * otomasyon_sonrasi_vardiya * 12  # Yıllık maliyet

    # Maliyet tasarrufu
    maliyet_tasarrufu = mevcut_maliyet - otomasyon_sonrasi_maliyet
    maliyet_sayfasi['B14'].value = maliyet_tasarrufu  # Yıllık Maliyet Tasarrufu

def verimlilik_artisi_hesapla(verimlilik_sayfasi):
    # Mevcut verimlilik
    mevcut_kapasite = verimlilik_sayfasi['B5'].value or 0
    mevcut_oee = verimlilik_sayfasi['B6'].value or 0
    mevcut_calisma_gunu = verimlilik_sayfasi['B7'].value or 0
    mevcut_uretim = mevcut_kapasite * mevcut_oee * mevcut_calisma_gunu

    # Otomasyon sonrası verimlilik
    otomasyon_sonrasi_kapasite = verimlilik_sayfasi['B10'].value or 0
    otomasyon_sonrasi_oee = verimlilik_sayfasi['B11'].value or 0
    otomasyon_sonrasi_calisma_gunu = verimlilik_sayfasi['B12'].value or 0
    otomasyon_sonrasi_uretim = otomasyon_sonrasi_kapasite * otomasyon_sonrasi_oee * otomasyon_sonrasi_calisma_gunu

    # Verimlilik artışı
    verimlilik_artisi = otomasyon_sonrasi_uretim - mevcut_uretim
    birim_urun_fiyati = verimlilik_sayfasi['B15'].value or 0
    verimlilik_geliri = verimlilik_artisi * birim_urun_fiyati
    verimlilik_sayfasi['B20'].value = verimlilik_geliri  # Yıllık Verimlilik Artışı

def kalite_iyilestirme_hesapla(kalite_sayfasi):
    # Mevcut kalite maliyeti
    mevcut_iade_sayisi = kalite_sayfasi['B5'].value or 0
    mevcut_iade_maliyeti = kalite_sayfasi['B6'].value or 0
    mevcut_sikayet_sayisi = kalite_sayfasi['B7'].value or 0
    mevcut_sikayet_maliyeti = kalite_sayfasi['B8'].value or 0
    mevcut_kalite_maliyeti = (mevcut_iade_sayisi * mevcut_iade_maliyeti) + (mevcut_sikayet_sayisi * mevcut_sikayet_maliyeti)

    # Otomasyon sonrası kalite maliyeti
    otomasyon_sonrasi_iade_sayisi = kalite_sayfasi['B11'].value or 0
    otomasyon_sonrasi_iade_maliyeti = kalite_sayfasi['B12'].value or 0
    otomasyon_sonrasi_sikayet_sayisi = kalite_sayfasi['B13'].value or 0
    otomasyon_sonrasi_sikayet_maliyeti = kalite_sayfasi['B14'].value or 0
    otomasyon_sonrasi_kalite_maliyeti = (otomasyon_sonrasi_iade_sayisi * otomasyon_sonrasi_iade_maliyeti) + (otomasyon_sonrasi_sikayet_sayisi * otomasyon_sonrasi_sikayet_maliyeti)

    # Kalite iyileştirme
    kalite_iyilestirme = mevcut_kalite_maliyeti - otomasyon_sonrasi_kalite_maliyeti
    kalite_sayfasi['B22'].value = kalite_iyilestirme  # Yıllık Kalite İyileştirme

def roi_hesapla(ozet_sayfasi, wb):
    # ROI hesaplamaları zaten Excel formülleriyle yapılıyor
    pass