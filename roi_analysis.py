import os
from glob import glob
from openpyxl import load_workbook


def find_latest_report(folder: str = ".") -> str:
    """Return path to the newest .xlsx file in *folder*.

    Parameters
    ----------
    folder : str, optional
        Directory to search for Excel files.

    Raises
    ------
    FileNotFoundError
        If no Excel files are found.
    """
    files = [f for f in glob(os.path.join(folder, "*.xlsx")) if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError("Klasörde .xlsx uzantılı dosya bulunamadı.")
    return max(files, key=os.path.getmtime)


def parse_metrics(path: str) -> dict:
    """Read ROI metrics from the summary sheet of *path*.

    Parameters
    ----------
    path : str
        Excel workbook path.

    Returns
    -------
    dict
        Dictionary with keys ``yatirim``, ``roi``, ``geri_odeme``, ``npv``.
    """
    wb = load_workbook(path, data_only=True)
    sheet = wb.get_sheet_by_name("4-Özet ve ROI") if hasattr(wb, "get_sheet_by_name") else wb["4-Özet ve ROI"]
    metrics = {
        "yatirim": sheet["B21"].value,
        "roi": sheet["B22"].value,
        "geri_odeme": sheet["B23"].value,
        "npv": sheet["B24"].value,
    }
    wb.close()
    return metrics


def yorum_yap(metrics: dict) -> str:
    """Return a textual recommendation based on ROI metrics."""
    yatirim = metrics.get("yatirim")
    roi = metrics.get("roi")
    geri_odeme = metrics.get("geri_odeme")
    npv = metrics.get("npv")

    if None in (yatirim, roi, geri_odeme, npv):
        return "Excel dosyasındaki veriler tam olarak okunamadı."

    mesaj = (
        f"Toplam Yatırım Maliyeti: {yatirim}\n"
        f"ROI: {roi:.2%}\n"
        f"Geri Ödeme Süresi: {geri_odeme:.1f} yıl\n"
        f"Net Bugünkü Değer: {npv}"
    )

    if roi > 0.5 and npv > 0 and geri_odeme <= 3:
        mesaj += "\n\nProjeye yatırım yapmak oldukça karlı görünüyor."
    elif roi > 0.1 and npv > 0:
        mesaj += "\n\nProjeye yatırım yapılabilir fakat detaylı analiz önerilir."
    else:
        mesaj += "\n\nFinansal göstergeler yatırımı desteklemiyor."

    return mesaj


def main(folder: str = ".") -> None:
    """Locate the latest report and output a short recommendation."""
    try:
        path = find_latest_report(folder)
    except FileNotFoundError as exc:
        print(exc)
        return

    metrics = parse_metrics(path)
    print(yorum_yap(metrics))


if __name__ == "__main__":
    main()
