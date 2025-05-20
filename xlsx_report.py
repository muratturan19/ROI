"""Generate an ROI report using xlsxwriter with advanced formatting."""

import xlsxwriter
from datetime import datetime

from data_templates import (
    ozet_roi_data,
    maliyet_tasarrufu_data,
    verimlilik_artisi_data,
    kalite_iyilestirme_data,
    npv_roi_bilgi_data,
)
from formatting import (
    sayfa_bicimlendir_xlsxwriter,
    sutun_genislikleri_ayarla_xlsxwriter,
    grafik_ekle_xlsxwriter,
)


def create_xlsxwriter_report(data, filename="roi_report_xlsxwriter.xlsx"):
    """Create a workbook using xlsxwriter and apply advanced formatting.

    Parameters
    ----------
    data : dict
        Values collected from the GUI form.
    filename : str, optional
        Output workbook name.
    """
    workbook = xlsxwriter.Workbook(filename)
    workbook.add_worksheet("Grafikler")

    ana_sayfa_data = [
        ["Otomasyon ve Proses İyileştirme ROI Hesaplama Aracı"],
        [],
        ["Şirket Adı:", "", "🏢"],
        ["Proje Adı:", "", "📋"],
        ["Tarih:", datetime.now().strftime("%d.%m.%Y"), "📅"],
        [],
        ["Bu araç, otomasyon ve proses iyileştirme projelerinin finansal etkisini hesaplamak için tasarlanmıştır."],
        [],
        ["İçindekiler:"],
        ["1. Maliyet Tasarrufu Hesaplama", "💰"],
        ["2. Verimlilik Artışı Hesaplama", "📈"],
        ["3. Kalite İyileştirme Hesaplama", "🔍"],
        ["4. Özet ve ROI Analizi", "📊"],
        ["5. NPV ve ROI Bilgi", "ℹ️"],
    ]

    templates = [
        ("Ana Sayfa", ana_sayfa_data),
        ("1-Maliyet Tasarrufu", maliyet_tasarrufu_data),
        ("2-Verimlilik Artışı", verimlilik_artisi_data),
        ("3-Kalite İyileştirme", kalite_iyilestirme_data),
        ("4-Özet ve ROI", ozet_roi_data),
        ("5-NPV_ROI Bilgi", npv_roi_bilgi_data),
    ]

    for sheet_name, tpl in templates:
        sheet = workbook.add_worksheet(sheet_name)
        for r, row in enumerate(tpl):
            sheet.write_row(r, 0, row)
        sutun_genislikleri_ayarla_xlsxwriter(sheet, workbook)
        sayfa_bicimlendir_xlsxwriter(sheet, workbook)

    # Ana sayfa bilgileri
    ana = workbook.get_worksheet_by_name("Ana Sayfa")
    if ana:
        ana.write("B3", data.get("sirket_adi", ""))
        ana.write("B4", data.get("proje_adi", ""))

    # Maliyet Tasarrufu sayfası verileri
    maliyet = workbook.get_worksheet_by_name("1-Maliyet Tasarrufu")
    if maliyet:
        maliyet.write_number("B5", data.get("mevcut_isci_sayisi", 0))
        maliyet.write_number("B6", data.get("ortalama_maas", 0))
        maliyet.write_number("B7", data.get("mevcut_vardiya_sayisi", 0))
        maliyet.write_number("B11", data.get("otomasyon_sonrasi_isci_sayisi", 0))
        maliyet.write_number("B12", data.get("otomasyon_sonrasi_maas", 0))
        maliyet.write_number("B13", data.get("otomasyon_sonrasi_vardiya_sayisi", 0))

        mevcut_maliyet = (
            data.get("mevcut_isci_sayisi", 0)
            * data.get("ortalama_maas", 0)
            * data.get("mevcut_vardiya_sayisi", 0)
            * 12
        )
        otomasyon_maliyet = (
            data.get("otomasyon_sonrasi_isci_sayisi", 0)
            * data.get("otomasyon_sonrasi_maas", 0)
            * data.get("otomasyon_sonrasi_vardiya_sayisi", 0)
            * 12
        )
        maliyet_tasarrufu = mevcut_maliyet - otomasyon_maliyet
        maliyet.write_number("B14", maliyet_tasarrufu)
    else:
        maliyet_tasarrufu = 0

    # Verimlilik artışı sayfası verileri
    verim = workbook.get_worksheet_by_name("2-Verimlilik Artışı")
    if verim:
        verim.write_number("B5", data.get("max_kapasite", 0))
        verim.write_number("B6", data.get("oee_mevcut", 0) / 100)
        verim.write_number("B7", data.get("calisma_gunu", 0))
        verim.write_number("B10", data.get("otomasyon_sonrasi_max_kapasite", 0))
        verim.write_number("B11", data.get("otomasyon_sonrasi_oee", 0) / 100)
        verim.write_number("B12", data.get("otomasyon_sonrasi_calisma_gunu", 0))
        verim.write_number("B15", data.get("birim_urun_fiyati", 0))

        mevcut_uretim = (
            data.get("max_kapasite", 0)
            * (data.get("oee_mevcut", 0) / 100)
            * data.get("calisma_gunu", 0)
        )
        otomasyon_uretim = (
            data.get("otomasyon_sonrasi_max_kapasite", 0)
            * (data.get("otomasyon_sonrasi_oee", 0) / 100)
            * data.get("otomasyon_sonrasi_calisma_gunu", 0)
        )
        verim_artisi = (otomasyon_uretim - mevcut_uretim) * data.get("birim_urun_fiyati", 0)
        verim.write_number("B20", verim_artisi)
    else:
        verim_artisi = 0

    # Kalite iyileştirme sayfası verileri
    kalite = workbook.get_worksheet_by_name("3-Kalite İyileştirme")
    if kalite:
        kalite.write_number("B5", data.get("iade_urun_sayisi", 0))
        kalite.write_number("B6", data.get("ortalama_iade_maliyeti", 0))
        kalite.write_number("B7", data.get("musteri_sikayet_sayisi", 0))
        kalite.write_number("B8", data.get("ortalama_sikayet_maliyeti", 0))
        kalite.write_number("B12", data.get("otomasyon_sonrasi_iade_urun_sayisi", 0))
        kalite.write_number("B13", data.get("otomasyon_sonrasi_iade_maliyeti", 0))
        kalite.write_number("B14", data.get("otomasyon_sonrasi_sikayet_sayisi", 0))
        kalite.write_number("B15", data.get("otomasyon_sonrasi_sikayet_maliyeti", 0))

        mevcut_kalite = (
            data.get("iade_urun_sayisi", 0) * data.get("ortalama_iade_maliyeti", 0)
            + data.get("musteri_sikayet_sayisi", 0) * data.get("ortalama_sikayet_maliyeti", 0)
        )
        otomasyon_kalite = (
            data.get("otomasyon_sonrasi_iade_urun_sayisi", 0) * data.get("otomasyon_sonrasi_iade_maliyeti", 0)
            + data.get("otomasyon_sonrasi_sikayet_sayisi", 0) * data.get("otomasyon_sonrasi_sikayet_maliyeti", 0)
        )
        kalite_gelir = mevcut_kalite - otomasyon_kalite
        kalite.write_formula("B19", "=(B5*B6+B7*B8)-(B12*B13+B14*B15)")
    else:
        kalite_gelir = 0

    summary = workbook.get_worksheet_by_name("4-Özet ve ROI")
    if summary:
        summary.write_number("B15", maliyet_tasarrufu)
        summary.write_number("B16", verim_artisi)
        summary.write_number("B17", kalite_gelir)
        summary.write_formula("B18", "=SUM(B15:B17)")
        summary.write_formula("B11", "=SUM(B5:B9)")
        summary.write_formula("B21", "=B11")
        summary.write_formula("B22", "=IF(B11>0,B18/B11,0)")
        summary.write_formula("B23", "=IF(B18>0,B11/B18,0)")

        for i in range(5):
            summary.write(i + 3, 3, i + 1)

        summary.write_formula("E4", "=B18")
        summary.write_formula("E5", "=E4*(1+B34)")
        summary.write_formula("E6", "=E5*(1+B34)")
        summary.write_formula("E7", "=E6*(1+B34)")
        summary.write_formula("E8", "=E7*(1+B34)")
        summary.write_formula("F4", "=E4")
        summary.write_formula("F5", "=E5/(1+B32)^1")
        summary.write_formula("F6", "=E6/(1+B32)^2")
        summary.write_formula("F7", "=E7/(1+B32)^3")
        summary.write_formula("F8", "=E8/(1+B32)^4")
        summary.write_formula("G4", "=E4")
        for r in range(5, 9):
            summary.write_formula(f"G{r}", f"=G{r-1}+E{r}")
        for r in range(4, 9):
            summary.write_formula(f"H{r}", "=B11")
        summary.write_formula("B24", "=SUM(F4:INDEX(F4:F8,B33))-B11")
        summary.write_formula("B27", "=IF(B11>0,B11*(1+B32)^B33,0)")
        summary.write_formula(
            "B28",
            "=IF(B11>0,B27/(1+B31)^B32-B11,0)",
        )

        # Helper cells for NPV comparison chart now stored on "Grafikler" sheet
        charts = workbook.get_worksheet_by_name("Grafikler")
        if charts:
            charts.write("A1", "Otomasyon NPV")
            charts.write_formula("B1", "='4-Özet ve ROI'!B24")
            charts.write("A2", "Banka NPV")
            charts.write_formula("B2", "='4-Özet ve ROI'!B28")

        summary.conditional_format("B22", {"type": "3_color_scale"})
        summary.conditional_format("B23", {"type": "data_bar", "bar_color": "#63C384"})

    grafik_ekle_xlsxwriter(workbook, "Grafikler")
    workbook.close()


if __name__ == "__main__":
    create_xlsxwriter_report({})
