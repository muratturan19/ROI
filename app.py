from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Otomasyon ROI & NPV Aracı",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

st.markdown(
    """
<style>
    /* Ana tema renkleri */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
    }

    /* Özel metric kartları */
    .metric-card {
        background: linear-gradient(135deg, var(--card-background) 0%, #f1f3f4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }

    .metric-title {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }

    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Başlık stilleri */
    .main-title {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Sidebar stilleri */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Kart container */
    .dashboard-container {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }

    /* Responsive grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    /* Chart container */
    .chart-container {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

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


def create_advanced_charts(df, summary, I, N):
    """Gelişmiş grafik setini oluşturur"""

    fig_main = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Kümülatif Net Kazanç",
            "Aylık Fayda Dağılımı",
            "Nakit Akışı Trendi",
            "Fayda Kategorileri",
        ),
        specs=[[{"secondary_y": True}, {"type": "bar"}], [{"secondary_y": True}, {"type": "pie"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    fig_main.add_trace(
        go.Scatter(
            x=df["Ay"],
            y=df["Kümülatif Net"],
            mode="lines+markers",
            name="Kümülatif Net",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
            hovertemplate="<b>Ay %{x}</b><br>Kümülatif Net: %{y:,.0f} TL<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig_main.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Başabaş Noktası",
        row=1,
        col=1,
    )

    fig_main.add_trace(
        go.Bar(
            x=df["Ay"],
            y=df["İşçilik Tasarrufu"],
            name="İşçilik",
            marker_color="#2ca02c",
        ),
        row=1,
        col=2,
    )

    fig_main.add_trace(
        go.Bar(
            x=df["Ay"],
            y=df["Hurda/Hata Tasarrufu"],
            name="Hurda/Hata",
            marker_color="#ff7f0e",
        ),
        row=1,
        col=2,
    )

    fig_main.add_trace(
        go.Bar(
            x=df["Ay"],
            y=df["Zaman Tasarrufu"],
            name="Zaman",
            marker_color="#d62728",
        ),
        row=1,
        col=2,
    )

    fig_main.add_trace(
        go.Bar(
            x=df["Ay"],
            y=df["Gelir Artışı"],
            name="Gelir Artışı",
            marker_color="#9467bd",
        ),
        row=1,
        col=2,
    )

    colors = ["green" if x > 0 else "red" for x in df["Nakit Akışı (CF)"]]
    fig_main.add_trace(
        go.Bar(
            x=df["Ay"],
            y=df["Nakit Akışı (CF)"],
            name="Nakit Akışı",
            marker_color=colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    categories = ["İşçilik", "Hurda/Hata", "Zaman", "Gelir Artışı"]
    values = [
        df["İşçilik Tasarrufu"].sum(),
        df["Hurda/Hata Tasarrufu"].sum(),
        df["Zaman Tasarrufu"].sum(),
        df["Gelir Artışı"].sum(),
    ]

    fig_main.add_trace(
        go.Pie(
            labels=categories,
            values=values,
            name="Fayda Dağılımı",
            hole=0.4,
            marker_colors=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"],
        ),
        row=2,
        col=2,
    )

    fig_main.update_layout(
        height=800,
        showlegend=True,
        title_text="ROI Analizi Dashboard",
        title_x=0.5,
        title_font_size=24,
        font=dict(size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig_main.update_traces(row=1, col=2, selector=dict(type="bar"))
    fig_main.update_xaxes(title_text="Ay", row=2, col=1)
    fig_main.update_yaxes(title_text="TL", row=2, col=1)

    return fig_main


def create_comparison_chart(summary, I):
    """Karşılaştırma grafiği"""
    fig_comp = go.Figure()

    categories = ["Yatırım", "Toplam Fayda", "NPV", "Alternatif Getiri"]
    values = [
        -I,
        summary["Toplam Aylık Fayda"],
        summary["NPV"],
        summary["FV Alternatif"],
    ]

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]

    fig_comp.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:,.0f} TL" for v in values],
            textposition="auto",
        )
    )

    fig_comp.update_layout(
        title="Yatırım Karşılaştırması",
        title_x=0.5,
        xaxis_title="Kategori",
        yaxis_title="Tutar (TL)",
        showlegend=False,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig_comp


def create_sensitivity_analysis():
    """Duyarlılık analizi için placeholder"""
    pass

# -------- UI --------
with st.sidebar:
    st.markdown(
        """
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 1rem;">
    <h2 style="margin: 0; color: white;">⚙️ Parametreler</h2>
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Yatırım analizini özelleştirin</p>
</div>
""",
        unsafe_allow_html=True,
    )

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
        opex_pm = st.number_input(
            "Aylık Opex (TL) – ROI tanımında kullanılmaz, NPV'de CF'ten düşülür",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
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
    I=I,
    N=N,
    r_annual=r_annual,
    E=E,
    C0=C0,
    i_annual=i_annual,
    s_before=s_before,
    s_after=s_after,
    units_per_month=units_pm,
    defect_cost=defect_cost,
    hours_saved_pm=hours_saved_pm,
    hourly_rate=hourly_rate,
    revenue_base_pm=revenue_base_pm,
    uplift_pct=uplift_pct,
    opex_pm=opex_pm,
)

df = res["df"]
summary = res["summary"]

if "I" in locals() and I > 0:
    total_benefit = summary.get("Toplam Aylık Fayda", 0)
    progress = min(total_benefit / I, 2.0) if I > 0 else 0
    st.sidebar.progress(progress / 2, text=f"Fayda/Yatırım Oranı: {progress:.1%}")

fig_main = create_advanced_charts(df, summary, I, N)
fig_comp = create_comparison_chart(summary, I)

st.markdown('<h1 class="main-title">🚀 Otomasyon ROI & NPV Aracı</h1>', unsafe_allow_html=True)

st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    payback_value = summary.get("Payback (ay)")
    payback_color = "🟢" if payback_value and payback_value <= 24 else "🟡" if payback_value and payback_value <= 36 else "🔴"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">{payback_color} Geri Dönüş Süresi</div>
        <div class="metric-value">{payback_value if payback_value else '∞'}</div>
        <div style="color: gray;">ay</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    roi_value = summary.get("ROI (N ay) %", 0.0)
    roi_color = "🟢" if roi_value > 50 else "🟡" if roi_value > 20 else "🔴"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">{roi_color} ROI ({N} ay)</div>
        <div class="metric-value">{roi_value:.1f}%</div>
        <div style="color: gray;">{summary['Toplam Aylık Fayda']:,.0f} TL toplam fayda</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    npv_value = summary.get("NPV", 0.0)
    npv_color = "🟢" if npv_value > 0 else "🔴"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">{npv_color} Net Bugünkü Değer</div>
        <div class="metric-value">{npv_value:,.0f}</div>
        <div style="color: gray;">TL</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    fv_diff = summary.get("FV Proje", 0.0) - summary.get("FV Alternatif", 0.0)
    fv_color = "🟢" if fv_diff > 0 else "🔴"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">{fv_color} Alternatife Göre Fark</div>
        <div class="metric-value">{fv_diff:,.0f}</div>
        <div style="color: gray;">TL ({N} ay sonunda)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig_main, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Detaylı Veri Tablosu")

    def color_negative_red(val):
        color = "red" if val < 0 else "green" if val > 0 else "black"
        return f"color: {color}"

    styled_df = df.style.format(
        {
            "Aylık Fayda": "{:,.0f}",
            "Nakit Akışı (CF)": "{:,.0f}",
            "Kümülatif Net": "{:,.0f}",
            "İşçilik Tasarrufu": "{:,.0f}",
            "Hurda/Hata Tasarrufu": "{:,.0f}",
            "Zaman Tasarrufu": "{:,.0f}",
            "Gelir Artışı": "{:,.0f}",
        }
    ).applymap(color_negative_red, subset=["Kümülatif Net", "Nakit Akışı (CF)"])

    st.dataframe(styled_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚖️ Yatırım Karşılaştırması")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("📈 Özet Analiz")

    if summary.get("NPV", 0) > 0:
        st.success(f"✅ Proje ekonomik olarak uygun! NPV: {summary['NPV']:,.0f} TL")
    else:
        st.error(f"❌ Proje ekonomik olarak uygun değil. NPV: {summary['NPV']:,.0f} TL")

    payback_value = summary.get("Payback (ay)")
    if payback_value and payback_value <= 24:
        st.success(f"⚡ Hızlı geri dönüş! {payback_value} ay")
    elif payback_value and payback_value <= 36:
        st.warning(f"⏳ Orta vadeli geri dönüş: {payback_value} ay")
    else:
        st.error("🐌 Uzun vadeli veya geri dönüş yok")

    if "amort_toggle" in locals() and amort_toggle:
        st.info(f"Muhasebe Görünümü: Aylık amortisman ~ {I/float(amort_months):,.0f} TL. (NPV'ye dahil edilmez)")

    st.markdown('</div>', unsafe_allow_html=True)
