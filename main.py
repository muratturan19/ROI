import sys
from gui import ROIHesaplamaArayuzu
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pencere = ROIHesaplamaArayuzu()
    pencere.show()
    sys.exit(app.exec_())
