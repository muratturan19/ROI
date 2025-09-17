import math
from typing import List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Otomasyon ROI & NPV Aracı", layout="wide")

# -------- Helpers --------

def monthly_rate_from_annual(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1/12) - 1

@st.cache_data(show_spinner=False)
def compute_series(
    I: float,
    N: int,
    r_annual: float,
    # Labor
    E: float,
    C0: float,
    i_annual: float,
    # Scrap/Defect
    s_before: float,
    s_after: float,
    units_per_month: float,
    defect_cost: float,
    # Time saving
    hours_saved_pm: float,
    hourly_rate: float,
    # Revenue uplift
    revenue_base_pm: float,
    uplift_pct: float,
    # Advanced
    opex_pm: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    r_m = monthly_rate_from_annual(r_annual)
    g_m = monthly_rate_from_annual(i_annual)

    months = np.arange(1, N+1)

    # Labor savings with monthly wage growth
    C_t = C0 * (1 + g_m) ** (months - 1)
    labor_saving = E * C_t

    # Scrap/defect savings (assumed constant monthly)
    scrap_saving = (s_before - s_after) * units_per_month * defect_cost
    scrap_series = np.full(N, scrap_saving)

    # Time saving (constant monthly)
    time_series = np.full(N, hours_saved_pm * hourly_rate)

    # Revenue uplift (constant monthly)
    revenue_series = np.full(N, revenue_base_pm * uplift_pct)

    monthly_benefit = labor_saving + scrap_series + time_series + revenue_series

    # Cash flow: exclude amortization; include optional opex
    cf = monthly_benefit - opex_pm

    # Cumulative net (with initial at t0)
    cum = -I + np.cumsum(cf)

    # NPV (monthly discount)
    discount_factors = (1 + r_m) ** months
    npv = -I + np.sum(cf / discount_factors)

    # FV comparison at month N (alternative vs project)
    fv_project = -I * (1 + r_m) ** N + np.sum(cf * (1 + r_m) ** (N - months))
    fv_alt = I * (1 + r_m) ** N

    # ROI (simple, no opex by definition): use monthly_benefit sum only
    roi_simple = ((monthly_benefit.sum() - I) / I) * 100 if I > 0 else float('nan')

    # Payback month
    payback_month = next((int(m) for m, v in zip(months, cum) if v >= 0), None)

    df = pd.DataFrame({
        "Ay": months,
        "Aylık Fayda": monthly_benefit,
        "Nakit Akışı (CF)": cf,
        "Kümülatif Net": cum,
        "İşçilik Tasarrufu": labor_saving,
        "Hurda/Hata Tasarrufu": scrap_series,
        "Zaman Tasarrufu": time_series,
        "Gelir Artışı": revenue_series,
    })

    summary = {
        "ROI (N ay) %": roi_simple,
        "NPV": npv,
        "Payback (ay)": payback_month,
        "FV Proje": fv_project,
        "FV Alternatif": fv_alt,
        "Toplam Aylık Fayda": monthly_benefit.sum(),
    }

    return {"df": df, "summary": summary}

# -------- UI --------
st.title("Otomasyon ROI & NPV Aracı")

with st.sidebar:
    st.header("Girdiler")

    st.subheader("Genel")
    I = st.number_input("Başlangıç Yatırımı (I)", min_value=0.0, value=1_000_000.0, step=50_000.0, format="%f")
    N = st.number_input("Horizon (Ay)", min_value=1, value=36, step=1)
    r_annual = st.number_input("İskonto / Alternatif Getiri (Yıllık %)", min_value=0.0, value=30.0, step=1.0) / 100.0

    st.subheader("İşçilik")
    E = st.number_input("Tasarruf Edilen İşçi Sayısı (E)", min_value=0.0, value=3.0, step=1.0)
    C0 = st.number_input("Bir İşçinin Bugünkü Aylık Maliyeti (C0)", min_value=0.0, value=50_000.0, step=1_000.0)
    i_annual = st.number_input("Maaş Artış Oranı (Yıllık %)", min_value=0.0, value=40.0, step=1.0) / 100.0

    st.subheader("Hurda / Hata")
    s_before = st.number_input("Hurda/Hata Oranı (Öncesi %)", min_value=0.0, max_value=100.0, value=5.0, step=0.1) / 100.0
    s_after = st.number_input("Hurda/Hata Oranı (Sonrası %)", min_value=0.0, max_value=100.0, value=2.0, step=0.1) / 100.0
    units_pm = st.number_input("Aylık Üretim Adedi", min_value=0.0, value=10_000.0, step=100.0)
    defect_cost = st.number_input("Bir Hatanın Maliyeti (TL)", min_value=0.0, value=250.0, step=10.0)

    st.subheader("Zaman Tasarrufu")
    hours_saved_pm = st.number_input("Aylık Zaman Kazancı (saat)", min_value=0.0, value=120.0, step=1.0)
    hourly_rate = st.number_input("Saatlik Ücret (TL)", min_value=0.0, value=400.0, step=10.0)

    st.subheader("Gelir Artışı (Opsiyonel)")
    revenue_base_pm = st.number_input("Baz Aylık Gelir (TL)", min_value=0.0, value=0.0, step=1_000.0)
    uplift_pct = st.number_input("Gelir Artış Oranı (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5) / 100.0

    with st.expander("İleri Düzey (Opex & Amortisman)"):
        opex_pm = st.number_input("Aylık Opex (TL) – ROI tanımında kullanılmaz, NPV'de CF'ten düşülür", min_value=0.0, value=0.0, step=100.0)
        amort_toggle = st.checkbox("Muhasebe Görünümü: Amortisman Göster")
        amort_months = st.number_input("Amortisman Süresi (Ay)", min_value=1, value=36, step=1)

    st.subheader("Senaryolar")
    if st.button("Küçük Atölye"):
        E, C0, i_annual = 2.0, 45_000.0, 0.30
        s_before, s_after, units_pm, defect_cost = 0.04, 0.02, 6000.0, 200.0
        hours_saved_pm, hourly_rate = 80.0, 300.0
        revenue_base_pm, uplift_pct = 0.0, 0.0
        I, N, r_annual = 600_000.0, 24, 0.28
    if st.button("Orta Ölçek"):
        E, C0, i_annual = 3.0, 50_000.0, 0.40
        s_before, s_after, units_pm, defect_cost = 0.05, 0.02, 10_000.0, 250.0
        hours_saved_pm, hourly_rate = 120.0, 400.0
        revenue_base_pm, uplift_pct = 0.0, 0.0
        I, N, r_annual = 1_000_000.0, 36, 0.30
    if st.button("Kurumsal"):
        E, C0, i_annual = 6.0, 60_000.0, 0.35
        s_before, s_after, units_pm, defect_cost = 0.06, 0.03, 25_000.0, 300.0
        hours_saved_pm, hourly_rate = 240.0, 500.0
        revenue_base_pm, uplift_pct = 1_500_000.0, 0.01
        I, N, r_annual = 3_000_000.0, 48, 0.26

# Compute
res = compute_series(
    I=I, N=N, r_annual=r_annual,
    E=E, C0=C0, i_annual=i_annual,
    s_before=s_before, s_after=s_after, units_per_month=units_pm, defect_cost=defect_cost,
    hours_saved_pm=hours_saved_pm, hourly_rate=hourly_rate,
    revenue_base_pm=revenue_base_pm, uplift_pct=uplift_pct,
    opex_pm=opex_pm,
)

df = res["df"]
summary = res["summary"]

# Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Yatırımın Geri Dönüş Süresi (ay)",
    summary['Payback (ay)'] if summary['Payback (ay)'] is not None else "—",
)
col2.metric(
    f"Yatırımın Toplam Getirisi ({N} ay)",
    f"{summary['ROI (N ay) %']:.1f} %",
)
col3.metric(
    f"Yatırımın Alternatif Getirisi ({N} ay)",
    f"{summary['FV Alternatif']:,.0f} TL",
)
col4.metric(
    "Projenin Gelecek Değeri",
    f"{summary['FV Proje']:,.0f} TL",
)

st.metric("Net Bugünkü Değer (NPV)", f"{summary['NPV']:,.0f} TL")

# Accounting view (amortization) card
if 'amort_toggle' in locals() and amort_toggle:
    st.info(f"Muhasebe Görünümü: Aylık amortisman ~ {I/float(amort_months):,.0f} TL. (NPV'ye dahil edilmez)")

# Chart: Cumulative Net
st.subheader("Kümülatif Net Kazanç (Ay)")
fig, ax = plt.subplots()
ax.plot(df["Ay"], df["Kümülatif Net"])
ax.set_xlabel("Ay")
ax.set_ylabel("Kümülatif Net (TL)")
st.pyplot(fig)

# Table
st.subheader("Ay Bazında Nakit Akışları")
st.dataframe(df.style.format({
    "Aylık Fayda": "{:,.0f}",
    "Nakit Akışı (CF)": "{:,.0f}",
    "Kümülatif Net": "{:,.0f}",
    "İşçilik Tasarrufu": "{:,.0f}",
    "Hurda/Hata Tasarrufu": "{:,.0f}",
    "Zaman Tasarrufu": "{:,.0f}",
    "Gelir Artışı": "{:,.0f}",
}))

st.caption(
    "Notlar: Toplam Getiri (ROI), yatırımın N ay boyunca sağladığı indirimsiz toplam kazanç / yatırım tutarıdır. "
    "Geri Dönüş Süresi, yatırımın kendini amorti ettiği ayı gösterir. ROI basit tanıma göre opex hariç hesaplanır. "
    "NPV nakit akışları ile aylık iskonto kullanır. Amortisman yalnızca muhasebe görünümünde bilgilendirme amaçlıdır."
)
