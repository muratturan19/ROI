from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QGroupBox,
    QStyle,
    QCheckBox,
)
from PyQt5.QtGui import QFont, QIcon, QDoubleValidator
from PyQt5.QtCore import Qt
import sys
from datetime import datetime
import logging
from data_templates import ozet_roi_data, maliyet_tasarrufu_data, verimlilik_artisi_data, kalite_iyilestirme_data, npv_roi_bilgi_data
from calculations import (
    maliyet_tasarrufu_hesapla,
    verimlilik_artisi_hesapla,
    kalite_iyilestirme_hesapla,
)
from xlsx_report import create_xlsxwriter_report
from formatting import get_common_styles

logging.basicConfig(level=logging.ERROR)

class ROIHesaplamaArayuzu(QMainWindow):
    """Main window providing forms to collect ROI parameters."""

    def __init__(self):
        """Initialize the window and build the interface."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """Create widgets and layout for all tabs."""
        self.setWindowTitle('ROI Hesaplama Aracı')
        self.setGeometry(100, 100, 800, 600)

        # Merkezi widget
        merkezi_widget = QWidget()
        self.setCentralWidget(merkezi_widget)

        # Ana düzen
        layout = QVBoxLayout()
        merkezi_widget.setLayout(layout)

        # Başlık
        baslik = QLabel('Otomasyon ROI Hesaplama Aracı')
        baslik.setFont(QFont('Calibri', 16, QFont.Bold))
        baslik.setAlignment(Qt.AlignCenter)
        layout.addWidget(baslik)

        # Tab Widget Oluştur
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)


        # Sayfaları Oluştur
        self.sirket_bilgileri_sayfasi()
        self.maliyet_tasarrufu_sayfasi()
        self.verimlilik_sayfasi()
        self.kalite_sayfasi()

        # Gelişmiş rapor seçeneği
        self.use_xlsxwriter = QCheckBox("Gelişmiş Excel Raporu (xlsxwriter)")
        self.use_xlsxwriter.setChecked(True)
        self.use_xlsxwriter.setToolTip(
            "İşaretli bırakıldığında grafik_ekle_xlsxwriter fonksiyonundaki dört grafikli rapor oluşturulur."
        )
        layout.addWidget(self.use_xlsxwriter)

        # Hesapla Butonu
        hesapla_btn = QPushButton('ROI Hesapla')
        hesapla_btn.setObjectName('calculateButton')
        hesapla_btn.clicked.connect(self.roi_hesapla)
        layout.addWidget(hesapla_btn)

        # Stil
        self.setStyleSheet(
            """
            QMainWindow { background-color: #F0F0F0; }
            QLabel { color: #4F81BD; padding: 5px; }
            QGroupBox { border: 1px solid #4F81BD; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; color: #4F81BD; font-weight: bold; }
            QLineEdit { background: #FFFFFF; padding: 3px; }
            QPushButton#calculateButton { background-color: #4F81BD; color: white; padding: 8px 12px; font-weight: bold; }
        """
        )

    def sirket_bilgileri_sayfasi(self):
        """Create the tab for entering company information."""
        sayfa = QWidget()
        ana_layout = QVBoxLayout()
        sayfa.setLayout(ana_layout)

        grup = QGroupBox("Şirket Bilgileri")
        layout = QFormLayout()

        # Şirket bilgileri input alanları
        self.sirket_adi = QLineEdit()
        self.proje_adi = QLineEdit()
        self.yetkili_kisi = QLineEdit()

        layout.addRow("Şirket Adı:", self.sirket_adi)
        layout.addRow("Proje Adı:", self.proje_adi)
        layout.addRow("Yetkili Kişi:", self.yetkili_kisi)

        grup.setLayout(layout)
        ana_layout.addWidget(grup)

        icon = self.style().standardIcon(QStyle.SP_FileDialogInfoView)
        self.tab_widget.addTab(sayfa, icon, "Şirket Bilgileri")

    def maliyet_tasarrufu_sayfasi(self):
        """Create the tab for labor cost comparison."""
        sayfa = QWidget()
        ana_layout = QVBoxLayout()
        sayfa.setLayout(ana_layout)

        mevcut_grup = QGroupBox("Mevcut Durum")
        mevcut_layout = QFormLayout()

        # Mevcut Durum Input Alanları
        self.mevcut_isci_sayisi = QLineEdit()
        self.mevcut_isci_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.ortalama_maas = QLineEdit()
        self.ortalama_maas.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.mevcut_vardiya_sayisi = QLineEdit()
        self.mevcut_vardiya_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))

        # Otomasyon Sonrası Input Alanları
        self.otomasyon_sonrasi_isci_sayisi = QLineEdit()
        self.otomasyon_sonrasi_isci_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_maas = QLineEdit()
        self.otomasyon_sonrasi_maas.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_vardiya_sayisi = QLineEdit()
        self.otomasyon_sonrasi_vardiya_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))

        # Mevcut Durum Grubu
        mevcut_layout.addRow("Mevcut İşçi Sayısı:", self.mevcut_isci_sayisi)
        mevcut_layout.addRow("Ortalama Aylık Maaş:", self.ortalama_maas)
        mevcut_layout.addRow("Vardiya Sayısı:", self.mevcut_vardiya_sayisi)
        mevcut_grup.setLayout(mevcut_layout)

        otomasyon_grup = QGroupBox("Otomasyon Sonrası")
        otomasyon_layout = QFormLayout()
        otomasyon_layout.addRow("İşçi Sayısı:", self.otomasyon_sonrasi_isci_sayisi)
        otomasyon_layout.addRow("Ortalama Aylık Maaş:", self.otomasyon_sonrasi_maas)
        otomasyon_layout.addRow("Vardiya Sayısı:", self.otomasyon_sonrasi_vardiya_sayisi)
        otomasyon_grup.setLayout(otomasyon_layout)

        ana_layout.addWidget(mevcut_grup)
        ana_layout.addWidget(otomasyon_grup)

        icon = self.style().standardIcon(QStyle.SP_DriveHDIcon)
        self.tab_widget.addTab(sayfa, icon, "Maliyet Tasarrufu")

    def verimlilik_sayfasi(self):
        """Create the tab for productivity inputs."""
        sayfa = QWidget()
        ana_layout = QVBoxLayout()
        sayfa.setLayout(ana_layout)

        mevcut_grup = QGroupBox("Mevcut Durum")
        mevcut_layout = QFormLayout()

        # Mevcut Durum Input Alanları
        self.max_kapasite = QLineEdit()
        self.max_kapasite.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.oee_mevcut = QLineEdit()
        self.oee_mevcut.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.calisma_gunu = QLineEdit()
        self.calisma_gunu.setValidator(QDoubleValidator(0.0, 366.0, 0))

        # Otomasyon Sonrası Input Alanları
        self.otomasyon_sonrasi_max_kapasite = QLineEdit()
        self.otomasyon_sonrasi_max_kapasite.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_oee = QLineEdit()
        self.otomasyon_sonrasi_oee.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.otomasyon_sonrasi_calisma_gunu = QLineEdit()
        self.otomasyon_sonrasi_calisma_gunu.setValidator(QDoubleValidator(0.0, 366.0, 0))

        # Birim Ürün Fiyatı
        self.birim_urun_fiyati = QLineEdit()
        self.birim_urun_fiyati.setValidator(QDoubleValidator(0.0, 1e12, 2))

        # Mevcut Durum Grubu
        mevcut_layout.addRow("Maksimum Günlük Kapasite (adet/gün):", self.max_kapasite)
        mevcut_layout.addRow("OEE (%) :", self.oee_mevcut)
        mevcut_layout.addRow("Yıllık Çalışma Günü:", self.calisma_gunu)
        mevcut_grup.setLayout(mevcut_layout)

        otomasyon_grup = QGroupBox("Otomasyon Sonrası")
        otomasyon_layout = QFormLayout()
        otomasyon_layout.addRow("Yeni Maksimum Kapasite (adet/gün):", self.otomasyon_sonrasi_max_kapasite)
        otomasyon_layout.addRow("Yeni OEE (%):", self.otomasyon_sonrasi_oee)
        otomasyon_layout.addRow("Yeni Yıllık Çalışma Günü:", self.otomasyon_sonrasi_calisma_gunu)
        otomasyon_grup.setLayout(otomasyon_layout)

        birim_grup = QGroupBox("Birim Ürün Fiyatı (TL/adet)")
        birim_layout = QFormLayout()
        birim_layout.addRow("Ürün Birim Fiyatı (TL):", self.birim_urun_fiyati)
        birim_grup.setLayout(birim_layout)

        ana_layout.addWidget(mevcut_grup)
        ana_layout.addWidget(otomasyon_grup)
        ana_layout.addWidget(birim_grup)

        icon = self.style().standardIcon(QStyle.SP_ArrowForward)
        self.tab_widget.addTab(sayfa, icon, "Verimlilik")

    def kalite_sayfasi(self):
        """Create the tab for quality metrics."""
        sayfa = QWidget()
        ana_layout = QVBoxLayout()
        sayfa.setLayout(ana_layout)

        mevcut_grup = QGroupBox("Mevcut Durum")
        mevcut_layout = QFormLayout()

        # Mevcut Durum Input Alanları
        self.iade_urun_sayisi = QLineEdit()
        self.iade_urun_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.ortalama_iade_maliyeti = QLineEdit()
        self.ortalama_iade_maliyeti.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.musteri_sikayet_sayisi = QLineEdit()
        self.musteri_sikayet_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.ortalama_sikayet_maliyeti = QLineEdit()
        self.ortalama_sikayet_maliyeti.setValidator(QDoubleValidator(0.0, 1e12, 2))

        # Otomasyon Sonrası Input Alanları
        self.otomasyon_sonrasi_iade_urun_sayisi = QLineEdit()
        self.otomasyon_sonrasi_iade_urun_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_iade_maliyeti = QLineEdit()
        self.otomasyon_sonrasi_iade_maliyeti.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_sikayet_sayisi = QLineEdit()
        self.otomasyon_sonrasi_sikayet_sayisi.setValidator(QDoubleValidator(0.0, 1e12, 2))
        self.otomasyon_sonrasi_sikayet_maliyeti = QLineEdit()
        self.otomasyon_sonrasi_sikayet_maliyeti.setValidator(QDoubleValidator(0.0, 1e12, 2))

        # Mevcut Durum Grubu
        mevcut_layout.addRow("Yıllık İade Ürün Sayısı:", self.iade_urun_sayisi)
        mevcut_layout.addRow("Ortalama İade Maliyeti:", self.ortalama_iade_maliyeti)
        mevcut_layout.addRow("Yıllık Müşteri Şikayet Sayısı:", self.musteri_sikayet_sayisi)
        mevcut_layout.addRow("Ortalama Şikayet Maliyeti:", self.ortalama_sikayet_maliyeti)
        mevcut_grup.setLayout(mevcut_layout)

        otomasyon_grup = QGroupBox("Otomasyon Sonrası")
        otomasyon_layout = QFormLayout()
        otomasyon_layout.addRow("Yıllık İade Ürün Sayısı:", self.otomasyon_sonrasi_iade_urun_sayisi)
        otomasyon_layout.addRow("Ortalama İade Maliyeti:", self.otomasyon_sonrasi_iade_maliyeti)
        otomasyon_layout.addRow("Yıllık Müşteri Şikayet Sayısı:", self.otomasyon_sonrasi_sikayet_sayisi)
        otomasyon_layout.addRow("Ortalama Şikayet Maliyeti:", self.otomasyon_sonrasi_sikayet_maliyeti)
        otomasyon_grup.setLayout(otomasyon_layout)

        ana_layout.addWidget(mevcut_grup)
        ana_layout.addWidget(otomasyon_grup)

        icon = self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        self.tab_widget.addTab(sayfa, icon, "Kalite")

    def _to_float(self, line_edit: QLineEdit) -> float:
        """Return the numeric value of a QLineEdit or zero.

        Displays a warning and returns 0.0 if conversion fails.
        """
        text = line_edit.text()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            QMessageBox.warning(self, "Geçersiz Giriş", f"'{text}' sayısal bir değer değil.")
            return 0.0

    def roi_hesapla(self):
        """Collect user inputs, perform calculations and export the report."""
        try:
            # Tüm input alanlarından verileri al
            sirket_adi = self.sirket_adi.text() or "Bilinmeyen Şirket"
            proje_adi = self.proje_adi.text() or "Isimsiz Proje"

            data = {
                "sirket_adi": sirket_adi,
                "proje_adi": proje_adi,
                "mevcut_isci_sayisi": self._to_float(self.mevcut_isci_sayisi),
                "ortalama_maas": self._to_float(self.ortalama_maas),
                "mevcut_vardiya_sayisi": self._to_float(self.mevcut_vardiya_sayisi),
                "otomasyon_sonrasi_isci_sayisi": self._to_float(self.otomasyon_sonrasi_isci_sayisi),
                "otomasyon_sonrasi_maas": self._to_float(self.otomasyon_sonrasi_maas),
                "otomasyon_sonrasi_vardiya_sayisi": self._to_float(self.otomasyon_sonrasi_vardiya_sayisi),
                "max_kapasite": self._to_float(self.max_kapasite),
                "oee_mevcut": self._to_float(self.oee_mevcut),
                "calisma_gunu": self._to_float(self.calisma_gunu),
                "otomasyon_sonrasi_max_kapasite": self._to_float(self.otomasyon_sonrasi_max_kapasite),
                "otomasyon_sonrasi_oee": self._to_float(self.otomasyon_sonrasi_oee),
                "otomasyon_sonrasi_calisma_gunu": self._to_float(self.otomasyon_sonrasi_calisma_gunu),
                "birim_urun_fiyati": self._to_float(self.birim_urun_fiyati),
                "iade_urun_sayisi": self._to_float(self.iade_urun_sayisi),
                "ortalama_iade_maliyeti": self._to_float(self.ortalama_iade_maliyeti),
                "musteri_sikayet_sayisi": self._to_float(self.musteri_sikayet_sayisi),
                "ortalama_sikayet_maliyeti": self._to_float(self.ortalama_sikayet_maliyeti),
                "otomasyon_sonrasi_iade_urun_sayisi": self._to_float(self.otomasyon_sonrasi_iade_urun_sayisi),
                "otomasyon_sonrasi_iade_maliyeti": self._to_float(self.otomasyon_sonrasi_iade_maliyeti),
                "otomasyon_sonrasi_sikayet_sayisi": self._to_float(self.otomasyon_sonrasi_sikayet_sayisi),
                "otomasyon_sonrasi_sikayet_maliyeti": self._to_float(self.otomasyon_sonrasi_sikayet_maliyeti),
            }

            dosya_adi = f"{sirket_adi}_{proje_adi}_ROI_Raporu_{datetime.now().strftime('%Y%m%d')}.xlsx"

            create_xlsxwriter_report(data, dosya_adi)

            QMessageBox.information(
                self,
                "Başarılı",
                f"ROI Raporu {dosya_adi} olarak kaydedildi!",
            )

        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Hesaplamada sorun oluştu: {str(e)}")

def main():
    """Launch the ROI calculator GUI."""
    app = QApplication(sys.argv)
    pencere = ROIHesaplamaArayuzu()
    pencere.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
