# calculations.py

def maliyet_tasarrufu_hesapla(maliyet_sayfasi):
    """Calculate yearly labor cost savings.

    Parameters
    ----------
    maliyet_sayfasi : openpyxl.worksheet.worksheet.Worksheet
        Worksheet containing cost inputs. The result is written to cell
        ``B14`` of this sheet.
    """
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
    """Compute increased production revenue after automation.

    Parameters
    ----------
    verimlilik_sayfasi : openpyxl.worksheet.worksheet.Worksheet
        Worksheet with productivity metrics. The calculated gain is stored in
        cell ``B20``.
    """
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
    """Determine savings from improved quality.

    Parameters
    ----------
    kalite_sayfasi : openpyxl.worksheet.worksheet.Worksheet
        Worksheet holding quality related costs. The computed improvement is
        placed in cell ``B19``.
    """
    # Mevcut kalite maliyeti
    mevcut_iade_sayisi = kalite_sayfasi['B5'].value or 0
    mevcut_iade_maliyeti = kalite_sayfasi['B6'].value or 0
    mevcut_sikayet_sayisi = kalite_sayfasi['B7'].value or 0
    mevcut_sikayet_maliyeti = kalite_sayfasi['B8'].value or 0
    mevcut_kalite_maliyeti = (mevcut_iade_sayisi * mevcut_iade_maliyeti) + (mevcut_sikayet_sayisi * mevcut_sikayet_maliyeti)

    # Otomasyon sonrası kalite maliyeti
    otomasyon_sonrasi_iade_sayisi = kalite_sayfasi['B12'].value or 0
    otomasyon_sonrasi_iade_maliyeti = kalite_sayfasi['B13'].value or 0
    otomasyon_sonrasi_sikayet_sayisi = kalite_sayfasi['B14'].value or 0
    otomasyon_sonrasi_sikayet_maliyeti = kalite_sayfasi['B15'].value or 0
    otomasyon_sonrasi_kalite_maliyeti = (otomasyon_sonrasi_iade_sayisi * otomasyon_sonrasi_iade_maliyeti) + (otomasyon_sonrasi_sikayet_sayisi * otomasyon_sonrasi_sikayet_maliyeti)

    # Kalite iyileştirme
    kalite_iyilestirme = mevcut_kalite_maliyeti - otomasyon_sonrasi_kalite_maliyeti
    kalite_sayfasi['B19'].value = kalite_iyilestirme  # Yıllık Kalite İyileştirme

def roi_hesapla(ozet_sayfasi, wb):
    """Calculate ROI related figures and update the summary sheet.

    This replicates the formulas that are inserted into the workbook so that the
    sheet contains actual numeric values.  The function reads the required
    inputs from ``ozet_sayfasi`` and writes the calculated results back to the
    same sheet.
    """

    # Toplam yatırım maliyeti (B5:B9)
    toplam_yatirim = sum(ozet_sayfasi[f"B{i}"].value or 0 for i in range(5, 10))
    # B11 hücresi formül içerdiğinden değeri doğrudan yazmayız

    # Yıllık getiriler
    maliyet_tasarrufu = ozet_sayfasi["B15"].value or 0
    verimlilik_artisi = ozet_sayfasi["B16"].value or 0
    kalite_iyilestirme = ozet_sayfasi["B17"].value or 0
    toplam_getiri = maliyet_tasarrufu + verimlilik_artisi + kalite_iyilestirme
    # B18 hücresi formülle hesaplanacak

    # Finansal analiz
    roi = (toplam_getiri / toplam_yatirim * 100) if toplam_yatirim else 0
    geri_odeme = (toplam_yatirim / toplam_getiri) if toplam_getiri else 0
    # B21, B22 ve B23 hücreleri formül içerdiğinden değer yazılmaz

    # Parametreler
    iskonto_orani = ozet_sayfasi["B32"].value or 0
    analiz_suresi = int(ozet_sayfasi["B33"].value or 0)
    maas_artisi = ozet_sayfasi["B34"].value or 0

    # Yıllık getirileri (E sütunu) ve bugünkü değerleri (F sütunu) hesapla
    yillik_getiriler = [toplam_getiri]
    for _ in range(1, 5):
        yillik_getiriler.append(yillik_getiriler[-1] * (1 + maas_artisi))

    bugunku_degerler = []
    for i, deger in enumerate(yillik_getiriler):
        ozet_sayfasi[f"E{i + 4}"].value = deger
        if i == 0:
            pv = deger
        else:
            pv = deger / ((1 + iskonto_orani) ** i)
        ozet_sayfasi[f"F{i + 4}"].value = pv
        bugunku_degerler.append(pv)

    # NPV hesapla (analiz süresi kadar yıl kullanılır)
    npv = sum(bugunku_degerler[:analiz_suresi]) - toplam_yatirim
    # B24 hücresi formülle hesaplanacak

    if toplam_yatirim > 0:
        banka_getiri = toplam_yatirim * ((1 + iskonto_orani) ** analiz_suresi)
        banka_npv = banka_getiri / ((1 + iskonto_orani) ** analiz_suresi) - sum(
            bugunku_degerler[:analiz_suresi]
        )
        # B27 ve B28 hücreleri formülle hesaplanacak
    else:
        banka_getiri = 0
        banka_npv = 0
