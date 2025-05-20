"""Generate an ROI report using xlsxwriter with advanced formatting."""

import xlsxwriter

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


def create_xlsxwriter_report(filename="roi_report_xlsxwriter.xlsx"):
    """Create a workbook using xlsxwriter and apply advanced formatting."""
    workbook = xlsxwriter.Workbook(filename)

    templates = [
        ("1-Maliyet Tasarrufu", maliyet_tasarrufu_data),
        ("2-Verimlilik Artışı", verimlilik_artisi_data),
        ("3-Kalite İyileştirme", kalite_iyilestirme_data),
        ("4-Özet ve ROI", ozet_roi_data),
        ("5-NPV_ROI Bilgi", npv_roi_bilgi_data),
    ]

    for sheet_name, data in templates:
        sheet = workbook.add_worksheet(sheet_name)
        for r, row in enumerate(data):
            sheet.write_row(r, 0, row)
        sutun_genislikleri_ayarla_xlsxwriter(sheet)
        sayfa_bicimlendir_xlsxwriter(sheet, workbook)

    summary = workbook.get_worksheet_by_name("4-Özet ve ROI")
    if summary:
        summary.write_formula("B11", "=SUM(B5:B9)")
        summary.write_formula("B18", "=SUM(B15:B17)")
        summary.write_formula("B21", "=B11")
        summary.write_formula("B22", "=IF(B11>0,(B18/B11)*100,0)")
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
        summary.write_formula("B24", "=SUM(F4:INDEX(F4:F8,B33))-B11")
        summary.write_formula("B27", "=IF(B11>0,B11*(1+B32)^B33,0)")
        summary.write_formula(
            "B28",
            "=IF(B11>0,B27/(1+B32)^B33-SUM(F4:INDEX(F4:F8,B33)),0)",
        )

        summary.conditional_format("B22", {"type": "3_color_scale"})
        summary.conditional_format("B23", {"type": "data_bar", "bar_color": "#63C384"})

    grafik_ekle_xlsxwriter(workbook)
    workbook.close()


if __name__ == "__main__":
    create_xlsxwriter_report()
