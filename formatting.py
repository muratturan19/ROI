def sayfa_bicimlendir_xlsxwriter(sheet, workbook):
    """Apply formatting rules for a worksheet created with xlsxwriter.

    Parameters
    ----------
    sheet : xlsxwriter.worksheet.Worksheet
        Worksheet to format.
    workbook : xlsxwriter.Workbook
        Parent workbook used to create style objects.
    """
    # Biçimlendirme formatları
    header_format = workbook.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': '#4F81BD',
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
        'bg_color': '#DDEBF7'  # Önemli satırları vurgulamak için açık mavi arka plan
    })
    percent_format = workbook.add_format({
        'border': 1,
        'num_format': '0.00%',  # Yüzde formatı
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'bold': True,
        'bg_color': '#DDEBF7'
    })
    year_format = workbook.add_format({
        'border': 1,
        'num_format': '0.00 "yıl"',  # Yıl formatı
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'bold': True,
        'bg_color': '#DDEBF7'
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
        sheet.set_column('D:D', 10)  # Yıl sütunu
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
    if sheet.name == "4-Özet ve ROI":
        sheet.set_column('C:C', 10)  # Grafikler için boşluk
        year_col_format = workbook.add_format({'num_format': '0'}) if workbook else None
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
            'value': True,
            'percentage': True,
            'leader_lines': True,
        },
    })
    chart1.set_title({'name': 'Proje Yatırım Maliyetleri'})
    chart1.set_style(10)
    chart1.set_size({'width': 400, 'height': 300})
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
    chart2.set_x_axis({'name': 'Yıl'})
    chart2.set_y_axis({'name': 'Bugünkü Değer (TL)'})
    chart2.set_size({'width': 350, 'height': 250})
    chart_sheet.insert_chart('B18', chart2)

    chart_cum = workbook.add_chart({'type': 'column'})
    chart_cum.add_series({
        'name': 'Kümülatif Getiri',
        'categories': ['4-Özet ve ROI', 3, 3, 7, 3],
        'values': ['4-Özet ve ROI', 3, 6, 7, 6],
    })
    chart_cum.set_y_axis({'name': 'Kümülatif Getiri (TL)'})

    chart_line = workbook.add_chart({'type': 'line'})
    chart_line.add_series({
        'name': 'Toplam Yatırım',
        'categories': ['4-Özet ve ROI', 3, 3, 7, 3],
        'values': ['4-Özet ve ROI', 3, 7, 7, 7],
    })
    chart_cum.combine(chart_line)
    chart_cum.set_title({'name': 'Kümülatif Getiri vs Yatırım'})
    chart_cum.set_size({'width': 350, 'height': 250})
    chart_sheet.insert_chart('B34', chart_cum)

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
    chart_npv.set_size({'width': 350, 'height': 250})
    chart_sheet.insert_chart('B50', chart_npv)

    # Highlight key performance indicators in a textbox
    kpi_text = (
        'ROI: =TEXT(B22,"0.00%")\n'
        'Geri Ödeme: =TEXT(B23,"0.0 \\"yıl\\"")\n'
        'Toplam Getiri: =TEXT(B18,"#,##0 ₺")\n'
        'NPV: =TEXT(B24,"#,##0 ₺")'
    )
    chart_sheet.insert_textbox(
        'B64',
        kpi_text,
        {'width': 260, 'height': 110, 'fill': {'color': '#DDEBF7'}, 'font': {'bold': True}}
    )

def roi_detay_hesapla(sheet, toplam_yatirim, toplam_getiri):
    """Placeholder for a detailed ROI calculation."""
    pass
