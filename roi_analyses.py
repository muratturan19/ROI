import os
import sys
from glob import glob
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTextEdit,
)
from PyQt5.QtCore import Qt

from roi_analysis import find_latest_report, parse_metrics, yorum_yap


class ROIAnalysesWindow(QWidget):
    """Simple interface to review generated ROI reports."""

    def __init__(self, folder: str = "."):
        super().__init__()
        self.folder = folder
        self.setWindowTitle("ROI Rapor Yorumlama")
        self.resize(500, 400)
        self.build_ui()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Excel raporu:"))

        self.combo = QComboBox()
        layout.addWidget(self.combo)

        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Yorumla")
        self.analyze_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

        self.refresh_files()
        # Automatically analyze the most recent report after the combo box
        # has been populated with available files and the newest one selected.
        self.run_analysis()

    def refresh_files(self) -> None:
        files = [f for f in glob(os.path.join(self.folder, "*.xlsx")) if os.path.isfile(f)]
        self.combo.clear()
        self.combo.addItems(files)
        if files:
            try:
                latest = find_latest_report(self.folder)
                index = files.index(latest)
            except Exception:
                index = 0
            self.combo.setCurrentIndex(index)

    def run_analysis(self) -> None:
        path = self.combo.currentText()
        if not path:
            self.text.setPlainText(".xlsx dosyası bulunamadı")
            return
        metrics = parse_metrics(path)
        result = yorum_yap(metrics)
        self.text.setPlainText(result)


def main() -> None:
    app = QApplication(sys.argv)
    window = ROIAnalysesWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
