def get_common_styles():
    """Return base color and font settings shared by both libraries."""
    return {
        "header_bg": "4F81BD",
        "subheader_bg": "B8CCE4",
        "highlight_bg": "DDEBF7",
        "border_color": "A0A0A0",
        "font_name": "Calibri",
    }


def sayfa_bicimlendir_xlsxwriter(sheet, workbook):
    """Apply formatting rules for a worksheet created with xlsxwriter.

    Parameters
    ----------
    sheet : xlsxwriter.worksheet.Worksheet
        Worksheet to format.
    workbook : xlsxwriter.Workbook
        Parent workbook used to create style objects.
    """
    styles = get_common_styles()

    # Biçimlendirme formatları
    header_format = workbook.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': f"#{styles['header_bg']}",
        'border': 1,
        'font_size': 12,
        'align': 'center',
        'valign': 'vcenter'
    })
    cell_format = workbook.add_format({
        'border': 1,
        'num_format': '#,##0.00 ₺',  # Para birimi formatı
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter'
    })
    number_format = workbook.add_format({
        'border': 1,
        'num_format': '#,##0.00',
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter'
    })
    bold_format = workbook.add_format({
        'bold': True,
        'border': 1,
        'num_format': '#,##0.00 ₺',
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'bg_color': f"#{styles['highlight_bg']}"  # Önemli satırları vurgulamak için açık mavi arka plan
    })
    percent_format = workbook.add_format({
        'border': 1,
        'num_format': '0.00%',  # Yüzde formatı
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'bold': True,
        'bg_color': f"#{styles['highlight_bg']}"
    })
    year_format = workbook.add_format({
        'border': 1,
        'num_format': '0.00 "yıl"',  # Yıl formatı
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'bold': True,
        'bg_color': f"#{styles['highlight_bg']}"
    })
    
    # Başlık satırını sabit olarak biçimlendir
    sheet.set_row(0, 30, header_format)  # İlk satır (başlık) için yükseklik ve biçimlendirme
    
    # Özet sayfası için özel biçimlendirme
    if sheet.name == "4-Özet ve ROI":
        # Önemli satırlara özel biçimlendirme
        sheet.set_row(3, None, bold_format)   # Proje Yatırım Maliyetleri
        sheet.set_row(9, None, bold_format)   # Toplam Proje Maliyeti
        sheet.set_row(13, None, bold_format)  # Yıllık Getiriler Özeti
        sheet.set_row(16, None, bold_format)  # Toplam Yıllık Getiri
        sheet.set_row(19, None, bold_format)  # Toplam Yatırım Maliyeti
        sheet.set_row(21, None, percent_format)  # ROI Değeri
        sheet.set_row(22, None, year_format)     # Geri Ödeme Süresi
        sheet.set_row(23, None, bold_format)     # Otomasyon Senaryosu NPV
        sheet.set_row(24, None, bold_format)     # Alternatif Senaryo başlığı
        sheet.set_row(25, None, bold_format)     # Bankaya Yatırmanın Analiz Süresi Sonu Değeri
        sheet.set_row(26, None, bold_format)     # Banka Senaryosu NPV
        sheet.set_row(28, None, percent_format)  # Faiz Oranı
        sheet.set_row(29, None, None)  # Analiz Süresi
        sheet.set_row(30, None, percent_format)  # Yıllık İşçilik Maaş Artışı
        
        # Yardımcı sütunlar için biçimlendirme
        year_col_format = workbook.add_format({'num_format': '0'})
        sheet.set_column('D:D', 10, year_col_format)  # Yıl sütunu
        sheet.set_column('E:E', 20, cell_format)  # Yıllık Getiri sütunu
        sheet.set_column('F:F', 20, cell_format)  # Bugünkü Değer sütunu
        
        # Diğer satırlara varsayılan biçimlendirme
        currency_rows = {4, 5, 6, 7, 8, 10, 14, 15, 16, 17, 20, 23, 26, 27}
        special_rows = {3, 9, 13, 16, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30}
        for row in range(1, 50):
            if row in special_rows:
                continue
            fmt = cell_format if row in currency_rows else number_format
            sheet.set_row(row, None, fmt)
    elif sheet.name == "5-NPV_ROI Bilgi":
        # NPV_ROI Bilgi sayfası için basit biçimlendirme
        sheet.set_row(0, 30, header_format)
        for row in range(1, 50):
            sheet.set_row(row, None, cell_format)
        sheet.set_column('A:A', 40)
        sheet.set_column('B:B', 20)
        sheet.set_column('C:C', 20)
        sheet.set_column('D:D', 20)
        sheet.set_column('E:E', 20)
    else:
        # Diğer sayfalar için varsayılan biçimlendirme
        for row in range(1, 50):
            sheet.set_row(row, None, bold_format if row in [3, 9, 13, 19] else number_format)

def sutun_genislikleri_ayarla_xlsxwriter(sheet, workbook=None):
    """Set column widths for the given xlsxwriter worksheet."""
    sheet.set_column('A:A', 40)  # A sütunu genişliği
    sheet.set_column('B:B', 20)  # B sütunu genişliği
    if sheet.name == "4-Özet ve ROI" and workbook:
        sheet.set_column('C:C', 10)  # Grafikler için boşluk
        year_col_format = workbook.add_format({'num_format': '0'})
        sheet.set_column('D:D', 10, year_col_format)  # Yıl sütunu
        sheet.set_column('E:E', 20)  # Yıllık Getiri sütunu
        sheet.set_column('F:F', 20)  # Bugünkü Değer sütunu

def grafik_ekle_xlsxwriter(workbook, chart_sheet_name="Grafikler"):
    """Insert summary charts into the specified worksheet."""
    ozet_sayfasi = workbook.get_worksheet_by_name("4-Özet ve ROI")
    chart_sheet = workbook.get_worksheet_by_name(chart_sheet_name)
    if not ozet_sayfasi or not chart_sheet:
        return
    
    # 1. Grafik: Proje Yatırım Maliyetleri (Pie Chart)
    chart1 = workbook.add_chart({'type': 'pie'})
    chart1.add_series({
        'name': 'Proje Yatırım Maliyetleri',
        'categories': ['4-Özet ve ROI', 3, 0, 7, 0],  # başlıklar
        'values': ['4-Özet ve ROI', 3, 1, 7, 1],      # değerler
        'data_labels': {
            'position': 'outside_end',
            'value': True,
            'percentage': True,
        },
    })
    chart1.set_title({'name': 'Proje Yatırım Maliyetleri'})
    chart1.set_style(10)
    chart1.set_size({'width': 700, 'height': 500})
    chart_sheet.insert_chart('B2', chart1)
    
    # 2. Grafik: Bugünkü Değerlerin Eğilimi (Line Chart)
    chart2 = workbook.add_chart({'type': 'line'})
    chart2.add_series({
        'name': 'Bugünkü Değer',
        'categories': ['4-Özet ve ROI', 3, 3, 7, 3],  # D4:D8 yıllar
        'values': ['4-Özet ve ROI', 3, 5, 7, 5],      # F4:F8 bugünkü değerler
        'line': {'color': '#C0504D'},
    })
    chart2.set_title({'name': 'NPV Eğilimi'})
    chart2.set_x_axis({'name': 'Yıl', 'num_format': '0'})
    chart2.set_y_axis({'name': 'Bugünkü Değer (TL)', 'num_format': '#,##0 ₺'})
    chart2.set_size({'width': 700, 'height': 500})
    chart_sheet.insert_chart('J2', chart2)

    chart_cum = workbook.add_chart({'type': 'column'})
    chart_cum.add_series({
        'name': 'Kümülatif Getiri',
        'categories': ['4-Özet ve ROI', 3, 3, 7, 3],
        'values': ['4-Özet ve ROI', 3, 6, 7, 6],
        'data_labels': {'value': True},
    })
    chart_cum.set_x_axis({'name': 'Yıl', 'num_format': '0'})
    chart_cum.set_y_axis({'name': 'Kümülatif Getiri (TL)', 'num_format': '#,##0 ₺'})

    chart_line = workbook.add_chart({'type': 'line'})
    chart_line.add_series({
        'name': 'Toplam Yatırım',
        'categories': ['4-Özet ve ROI', 3, 3, 7, 3],
        'values': ['4-Özet ve ROI', 3, 7, 7, 7],
        'line': {'color': 'red', 'width': 2}
    })
    chart_cum.combine(chart_line)
    chart_cum.set_title({'name': 'Kümülatif Getiri vs Yatırım'})
    chart_cum.set_size({'width': 700, 'height': 500})
    chart_sheet.insert_chart('B28', chart_cum)

    # 4. Grafik: Otomasyon ve Banka NPV Karşılaştırması
    chart_npv = workbook.add_chart({'type': 'column'})
    chart_npv.add_series({
        'name': 'Otomasyon NPV',
        'categories': [chart_sheet_name, 0, 0, 0, 0],  # A1
        'values':     [chart_sheet_name, 0, 1, 0, 1],  # B1
        'fill': {'color': '#63C384'},
    })
    chart_npv.add_series({
        'name': 'Banka NPV',
        'categories': [chart_sheet_name, 1, 0, 1, 0],  # A2
        'values':     [chart_sheet_name, 1, 1, 1, 1],  # B2
        'fill': {'color': '#BFBFBF'},
    })
    chart_npv.set_title({'name': 'NPV Karşılaştırması'})
    chart_npv.set_y_axis({'name': 'NPV (TL)'})
    chart_npv.set_size({'width': 700, 'height': 500})
    chart_sheet.insert_chart('J28', chart_npv)

    # Highlight key performance indicators in a textbox
    kpi_text = (
        'ROI: =TEXT(B22,"0.00%")\n'
        'Geri Ödeme: =TEXT(B23,"0.0 \\"yıl\\"")\n'
        'Toplam Getiri: =TEXT(B18,"#,##0 ₺")\n'
        'NPV: =TEXT(B24,"#,##0 ₺")'
    )
    chart_sheet.insert_textbox(
        'J54',
        kpi_text,
        {
            'width': 300,
            'height': 120,
            'fill': {'color': '#DDEBF7'},
            'font': {'bold': True},
            'align': {'horizontal': 'center', 'vertical': 'middle'},
        }
    )

def roi_detay_hesapla(sheet, toplam_yatirim, toplam_getiri):
    """Calculate detailed ROI metrics and write them to the worksheet.

    Parameters
    ----------
    sheet : Worksheet
        Either an :class:`xlsxwriter.worksheet.Worksheet` or an
        :class:`openpyxl.worksheet.worksheet.Worksheet` instance
        representing the summary sheet.
    toplam_yatirim : float
        Total investment amount.
    toplam_getiri : float
        First year return of the project.
    """

    # Default parameters used in the templates
    try:
        iskonto_orani = sheet["B32"].value or 0
        analiz_suresi = int(sheet["B33"].value or 0)
        maas_artisi = sheet["B34"].value or 0
    except Exception:
        # xlsxwriter worksheets do not support reading values, so fall back to
        # the constants defined in ``data_templates``.
        iskonto_orani = 0.4
        analiz_suresi = 3
        maas_artisi = 0.5

    # --- Yearly cash flows -------------------------------------------------
    yillik_getiriler = []
    bugunku_degerler = []
    kümülatif = []
    kum = 0

    for yil in range(5):
        if yil == 0:
            getir = float(toplam_getiri)
        else:
            getir = yillik_getiriler[-1] * (1 + maas_artisi)
        yillik_getiriler.append(getir)

        pv = getir / ((1 + iskonto_orani) ** yil)
        bugunku_degerler.append(pv)

        kum += getir
        kümülatif.append(kum)

    # Payback period based on cumulative cash flows
    geri_odeme = 0
    for idx, deger in enumerate(kümülatif, start=1):
        if deger >= toplam_yatirim:
            geri_odeme = idx
            break
    if geri_odeme == 0 and toplam_getiri:
        geri_odeme = toplam_yatirim / toplam_getiri

    roi = toplam_getiri / toplam_yatirim if toplam_yatirim else 0
    npv = sum(bugunku_degerler[:analiz_suresi]) - toplam_yatirim
    banka_getiri = toplam_yatirim * (1 + iskonto_orani) ** analiz_suresi
    banka_npv = banka_getiri / (1 + iskonto_orani) ** analiz_suresi - toplam_yatirim

    # Helper to write either with openpyxl or xlsxwriter
    def _write(cell, value):
        if hasattr(sheet, "write"):
            row = int(cell[1:]) - 1
            col = ord(cell[0].upper()) - ord("A")
            sheet.write_number(row, col, value)
        else:
            sheet[cell].value = value

    for idx in range(5):
        row = 4 + idx
        if hasattr(sheet, "write"):
            sheet.write_number(row - 1, 3, idx + 1)
            sheet.write_number(row - 1, 4, yillik_getiriler[idx])
            sheet.write_number(row - 1, 5, bugunku_degerler[idx])
            sheet.write_number(row - 1, 6, kümülatif[idx])
            sheet.write_number(row - 1, 7, toplam_yatirim)
        else:
            sheet.cell(row=row, column=4, value=idx + 1)
            sheet.cell(row=row, column=5, value=yillik_getiriler[idx])
            sheet.cell(row=row, column=6, value=bugunku_degerler[idx])
            sheet.cell(row=row, column=7, value=kümülatif[idx])
            sheet.cell(row=row, column=8, value=toplam_yatirim)

    _write("B22", roi)
    _write("B23", geri_odeme)
    _write("B24", npv)
    _write("B27", banka_getiri)
    _write("B28", banka_npv)

