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
        sheet.set_row(20, None, percent_format)  # ROI Değeri
        sheet.set_row(21, None, year_format)     # Geri Ödeme Süresi
        sheet.set_row(22, None, bold_format)     # Otomasyon Senaryosu NPV
        sheet.set_row(23, None, bold_format)     # Alternatif Senaryo başlığı
        sheet.set_row(24, None, bold_format)     # Bankaya Yatırmanın Analiz Süresi Sonu Değeri
        sheet.set_row(25, None, bold_format)     # Banka Senaryosu NPV
        sheet.set_row(28, None, percent_format)  # Faiz Oranı
        sheet.set_row(29, None, None)  # Analiz Süresi
        sheet.set_row(30, None, percent_format)  # Yıllık İşçilik Maaş Artışı
        
        # Yardımcı sütunlar için biçimlendirme
        sheet.set_column('D:D', 10)  # Yıl sütunu
        sheet.set_column('E:E', 20, cell_format)  # Yıllık Getiri sütunu
        sheet.set_column('F:F', 20, cell_format)  # Bugünkü Değer sütunu
        
        # Diğer satırlara varsayılan biçimlendirme
        for row in range(1, 50):
            if row not in [3, 9, 13, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30]:
                sheet.set_row(row, None, cell_format)
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
            sheet.set_row(row, None, bold_format if row in [3, 9, 13, 19] else cell_format)

def sutun_genislikleri_ayarla_xlsxwriter(sheet):
    """Set column widths for the given xlsxwriter worksheet."""
    sheet.set_column('A:A', 40)  # A sütunu genişliği
    sheet.set_column('B:B', 20)  # B sütunu genişliği
    if sheet.name == "4-Özet ve ROI":
        sheet.set_column('C:C', 10)  # Grafikler için boşluk
        sheet.set_column('D:D', 10)  # Yıl sütunu (zaten yukarıda ayarlandı)
        sheet.set_column('E:E', 20)  # Yıllık Getiri sütunu (zaten yukarıda ayarlandı)
        sheet.set_column('F:F', 20)  # Bugünkü Değer sütunu (zaten yukarıda ayarlandı)

def grafik_ekle_xlsxwriter(workbook):
    """Insert summary charts into the workbook."""
    ozet_sayfasi = workbook.get_worksheet_by_name("4-Özet ve ROI")
    
    # 1. Grafik: Proje Yatırım Maliyetleri (Pie Chart)
    chart1 = workbook.add_chart({'type': 'pie'})
    chart1.add_series({
        'name': 'Proje Yatırım Maliyetleri',
        'categories': ['4-Özet ve ROI', 3, 0, 7, 0],  # başlıklar
        'values': ['4-Özet ve ROI', 3, 1, 7, 1],      # değerler
        'data_labels': {'value': True, 'percentage': True},
    })
    chart1.set_title({'name': 'Proje Yatırım Maliyetleri'})
    chart1.set_style(10)
    ozet_sayfasi.insert_chart('H4', chart1)
    
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
    ozet_sayfasi.insert_chart('H20', chart2)

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
    ozet_sayfasi.insert_chart('H36', chart_cum)

def roi_detay_hesapla(sheet, toplam_yatirim, toplam_getiri):
    """Placeholder for a detailed ROI calculation."""
    pass
