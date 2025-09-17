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
    if I is None or (isinstance(I, (float, int, np.floating, np.integer)) and not np.isfinite(I)) or I <= 0:
        I = 1.0

    if N is None or N <= 0:
        N = 1

    if r_annual is None or (isinstance(r_annual, (float, int, np.floating, np.integer)) and not np.isfinite(r_annual)) or r_annual < 0:
        r_annual = 0.01

    if i_annual is None or (isinstance(i_annual, (float, int, np.floating, np.integer)) and not np.isfinite(i_annual)) or i_annual < 0:
        i_annual = 0.0

    N = int(N)

    r_m = monthly_rate_from_annual(float(r_annual))
    g_m = monthly_rate_from_annual(float(i_annual))

    months = np.arange(1, N + 1)

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
    total_benefit = float(np.sum(monthly_benefit))

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
    if I > 0 and total_benefit > 0:
        roi_simple = ((total_benefit - I) / I) * 100
    elif I > 0 and total_benefit == 0:
        roi_simple = -100.0
    elif I > 0 and total_benefit < 0:
        roi_simple = ((total_benefit - I) / I) * 100
    else:
        roi_simple = 0.0

    # Payback month
    if total_benefit > I:
        payback_month = next((int(m) for m, v in zip(months, cum) if v >= 0), None)
    else:
        payback_month = None

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
        "Toplam Aylık Fayda": total_benefit,
    }

    return {"df": df, "summary": summary}


def format_currency(amount, symbol="₺"):
    """Para formatlaması"""
    if amount is None:
        return "—"

    if isinstance(amount, (float, int, np.floating, np.integer)):
        if np.isinf(amount):
            return "∞"
        if isinstance(amount, (float, np.floating)) and np.isnan(amount):
            return "—"
    else:
        return f"{amount} {symbol}"

    if abs(amount) >= 1_000_000:
        return f"{amount/1_000_000:.1f}M {symbol}"
    elif abs(amount) >= 1_000:
        return f"{amount/1_000:.0f}K {symbol}"
    else:
        return f"{amount:,.0f} {symbol}"


def get_metric_color_and_icon(metric_type, value, threshold_good=None, threshold_medium=None):
    """Metrik için renk ve icon döndürür"""
    if metric_type == "payback":
        if value is None:
            return "🔴", "#d62728"
        elif value <= 12:
            return "🟢", "#2ca02c"
        elif value <= 24:
            return "🟡", "#ff7f0e"
        else:
            return "🔴", "#d62728"

    elif metric_type == "roi":
        if value >= 100:
            return "🚀", "#2ca02c"
        elif value >= 50:
            return "🟢", "#2ca02c"
        elif value >= 20:
            return "🟡", "#ff7f0e"
        else:
            return "🔴", "#d62728"

    elif metric_type == "npv":
        if value > 0:
            return "✅", "#2ca02c"
        else:
            return "❌", "#d62728"

    elif metric_type == "difference":
        if value > 0:
            return "📈", "#2ca02c"
        else:
            return "📉", "#d62728"

    else:
        return "📊", "#1f77b4"


def format_safe_metric(value, metric_type="general", currency_symbol="₺"):
    """Güvenli metrik formatlaması"""
    if value is None:
        return "—", "Hesaplanamıyor"

    if isinstance(value, (float, int, np.floating, np.integer)):
        if np.isinf(value):
            return "∞", "Sonsuz"
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "—", "Tanımsız"
    else:
        return f"{value}", ""

    if metric_type == "percentage":
        return f"{float(value):.1f}%", ""
    if metric_type == "currency":
        return format_currency(float(value), currency_symbol), ""
    if metric_type == "months":
        return f"{int(round(float(value)))} ay", ""

    return f"{float(value):,.0f}", ""


def create_advanced_charts(df, summary, I, N, currency_symbol="₺"):
    """Gelişmiş grafik setini oluşturur - para birimi desteği ile"""

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
            hovertemplate=f"<b>Ay %{{x}}</b><br>Kümülatif Net: %{{y:,.0f}} {currency_symbol}<extra></extra>",
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
    fig_main.update_yaxes(title_text=f"Tutar ({currency_symbol})", row=1, col=1)
    fig_main.update_yaxes(title_text=f"Tutar ({currency_symbol})", row=2, col=1)

    return fig_main


def create_comparison_chart(summary, I, currency_symbol="₺"):
    """Karşılaştırma grafiği - para birimi desteği ile"""
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
            text=[f"{format_currency(v, currency_symbol)}" for v in values],
            textposition="auto",
        )
    )

    fig_comp.update_layout(
        title="Yatırım Karşılaştırması",
        title_x=0.5,
        xaxis_title="Kategori",
        yaxis_title=f"Tutar ({currency_symbol})",
        showlegend=False,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig_comp


def create_sensitivity_analysis():
    """Duyarlılık analizi için placeholder"""
    pass

# -------- State Defaults --------
DEFAULT_PARAMS = {
    "I": 1_000_000.0,
    "r_annual": 30.0,
    "analysis_years": 3,
    "E": 3.0,
    "C0": 50_000.0,
    "i_annual": 40.0,
    "s_before": 5.0,
    "s_after": 2.0,
    "units_pm": 10_000.0,
    "defect_cost": 250.0,
    "hours_saved_pm": 120.0,
    "hourly_rate": 400.0,
    "revenue_base_pm": 0.0,
    "uplift_pct": 0.0,
    "opex_pm": 0.0,
    "amort_toggle": False,
    "amort_months": 36,
}

if "scenario_applied" not in st.session_state:
    st.session_state.scenario_applied = False

for param_key, param_value in DEFAULT_PARAMS.items():
    st.session_state.setdefault(param_key, param_value)

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
    I = st.number_input(
        "Başlangıç Yatırımı (I)",
        min_value=1.0,
        value=st.session_state.get("I", 1_000_000.0),
        step=50_000.0,
        format="%f",
        help="Minimum 1 TL olmalıdır",
    )
    st.session_state["I"] = I

    r_annual_pct = st.number_input(
        "İskonto / Alternatif Getiri (Yıllık %)",
        min_value=0.0,
        value=st.session_state.get("r_annual", 30.0),
        step=1.0,
    )
    st.session_state["r_annual"] = r_annual_pct
    r_annual = r_annual_pct / 100.0

    st.subheader("💰 Para Birimi")
    currency_options = {
        "TL": {"symbol": "₺", "name": "Türk Lirası"},
        "USD": {"symbol": "$", "name": "Amerikan Doları"},
        "EUR": {"symbol": "€", "name": "Euro"},
        "GBP": {"symbol": "£", "name": "İngiliz Sterlini"},
    }

    selected_currency = st.selectbox(
        "Para Birimi Seçin",
        options=list(currency_options.keys()),
        index=0,
        format_func=lambda x: f"{currency_options[x]['symbol']} {currency_options[x]['name']}",
    )

    currency_symbol = currency_options[selected_currency]["symbol"]
    currency_name = currency_options[selected_currency]["name"]

    st.subheader("📅 Analiz Dönemi")
    analysis_years = st.number_input(
        "Analiz Dönemi (Yıl)",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get("analysis_years", 3)),
        step=1,
        help="ROI ve getiri karşılaştırması için dönem",
    )
    analysis_years = int(analysis_years)
    st.session_state["analysis_years"] = analysis_years

    st.subheader("İşçilik")
    E = st.number_input(
        "Tasarruf Edilen İşçi Sayısı (E)",
        min_value=0.0,
        value=st.session_state.get("E", 3.0),
        step=1.0,
    )
    st.session_state["E"] = E

    C0 = st.number_input(
        "Bir İşçinin Bugünkü Aylık Maliyeti (C0)",
        min_value=0.0,
        value=st.session_state.get("C0", 50_000.0),
        step=1_000.0,
    )
    st.session_state["C0"] = C0

    i_annual_pct = st.number_input(
        "Maaş Artış Oranı (Yıllık %)",
        min_value=0.0,
        value=st.session_state.get("i_annual", 40.0),
        step=1.0,
    )
    st.session_state["i_annual"] = i_annual_pct
    i_annual = i_annual_pct / 100.0

    st.subheader("Hurda / Hata")
    s_before_pct = st.number_input(
        "Hurda/Hata Oranı (Öncesi %)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get("s_before", 5.0),
        step=0.1,
    )
    st.session_state["s_before"] = s_before_pct

    s_after_pct = st.number_input(
        "Hurda/Hata Oranı (Sonrası %)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get("s_after", 2.0),
        step=0.1,
    )
    st.session_state["s_after"] = s_after_pct

    s_before = s_before_pct / 100.0
    s_after = s_after_pct / 100.0

    units_pm = st.number_input(
        "Aylık Üretim Adedi",
        min_value=0.0,
        value=st.session_state.get("units_pm", 10_000.0),
        step=100.0,
    )
    st.session_state["units_pm"] = units_pm

    defect_cost = st.number_input(
        f"Bir Hatanın Maliyeti ({currency_symbol})",
        min_value=0.0,
        value=st.session_state.get("defect_cost", 250.0),
        step=10.0,
    )
    st.session_state["defect_cost"] = defect_cost

    st.subheader("Zaman Tasarrufu")
    hours_saved_pm = st.number_input(
        "Aylık Zaman Kazancı (saat)",
        min_value=0.0,
        value=st.session_state.get("hours_saved_pm", 120.0),
        step=1.0,
    )
    st.session_state["hours_saved_pm"] = hours_saved_pm

    hourly_rate = st.number_input(
        f"Saatlik Ücret ({currency_symbol})",
        min_value=0.0,
        value=st.session_state.get("hourly_rate", 400.0),
        step=10.0,
    )
    st.session_state["hourly_rate"] = hourly_rate

    st.subheader("Gelir Artışı (Opsiyonel)")
    revenue_base_pm = st.number_input(
        f"Baz Aylık Gelir ({currency_symbol})",
        min_value=0.0,
        value=st.session_state.get("revenue_base_pm", 0.0),
        step=1_000.0,
    )
    st.session_state["revenue_base_pm"] = revenue_base_pm

    uplift_pct_input = st.number_input(
        "Gelir Artış Oranı (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get("uplift_pct", 0.0),
        step=0.5,
    )
    st.session_state["uplift_pct"] = uplift_pct_input
    uplift_pct = uplift_pct_input / 100.0

    with st.expander("İleri Düzey (Opex & Amortisman)"):
        opex_pm = st.number_input(
            f"Aylık Opex ({currency_symbol}) – ROI tanımında kullanılmaz, NPV'de CF'ten düşülür",
            min_value=0.0,
            value=st.session_state.get("opex_pm", 0.0),
            step=100.0,
        )
        st.session_state["opex_pm"] = opex_pm

        amort_toggle = st.checkbox(
            "Muhasebe Görünümü: Amortisman Göster",
            value=st.session_state.get("amort_toggle", False),
        )
        st.session_state["amort_toggle"] = amort_toggle

        amort_months = st.number_input(
            "Amortisman Süresi (Ay)",
            min_value=1,
            value=int(st.session_state.get("amort_months", 36)),
            step=1,
        )
        amort_months = int(amort_months)
        st.session_state["amort_months"] = amort_months

    st.subheader("Senaryolar")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        if st.button("Küçük Atölye", use_container_width=True):
            st.session_state.update(
                {
                    "I": 600_000.0,
                    "E": 2.0,
                    "C0": 45_000.0,
                    "i_annual": 30.0,
                    "s_before": 4.0,
                    "s_after": 2.0,
                    "units_pm": 6_000.0,
                    "defect_cost": 200.0,
                    "hours_saved_pm": 80.0,
                    "hourly_rate": 300.0,
                    "revenue_base_pm": 0.0,
                    "uplift_pct": 0.0,
                    "analysis_years": 2,
                    "r_annual": 28.0,
                    "scenario_applied": True,
                }
            )
            st.rerun()

    with col_s2:
        if st.button("Orta Ölçek", use_container_width=True):
            st.session_state.update(
                {
                    "I": 1_000_000.0,
                    "E": 3.0,
                    "C0": 50_000.0,
                    "i_annual": 40.0,
                    "s_before": 5.0,
                    "s_after": 2.0,
                    "units_pm": 10_000.0,
                    "defect_cost": 250.0,
                    "hours_saved_pm": 120.0,
                    "hourly_rate": 400.0,
                    "revenue_base_pm": 0.0,
                    "uplift_pct": 0.0,
                    "analysis_years": 3,
                    "r_annual": 30.0,
                    "scenario_applied": True,
                }
            )
            st.rerun()

    with col_s3:
        if st.button("Kurumsal", use_container_width=True):
            st.session_state.update(
                {
                    "I": 3_000_000.0,
                    "E": 6.0,
                    "C0": 60_000.0,
                    "i_annual": 35.0,
                    "s_before": 6.0,
                    "s_after": 3.0,
                    "units_pm": 25_000.0,
                    "defect_cost": 300.0,
                    "hours_saved_pm": 240.0,
                    "hourly_rate": 500.0,
                    "revenue_base_pm": 1_500_000.0,
                    "uplift_pct": 1.0,
                    "analysis_years": 4,
                    "r_annual": 26.0,
                    "scenario_applied": True,
                }
            )
            st.rerun()

    s_before_diff = s_before - s_after
    total_monthly_benefit = float(
        (E * C0)
        + (s_before_diff * units_pm * defect_cost)
        + (hours_saved_pm * hourly_rate)
        + (revenue_base_pm * uplift_pct)
    )

    if total_monthly_benefit <= 0:
        st.sidebar.error("⚠️ Toplam aylık fayda sıfır! En az bir kalemde pozitif değer girin.")
        st.sidebar.info(
            "💡 İpucu: İşçilik, hurda azaltma, zaman tasarrufu veya gelir artışından en az birini aktifleştirin."
        )

# Dönem hesaplaması
analysis_years_value = int(analysis_years)
analysis_months = analysis_years_value * 12
N = analysis_months

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
    progress_ratio = total_benefit / I if I else 0.0
    if isinstance(progress_ratio, (float, int, np.floating, np.integer)) and np.isfinite(progress_ratio):
        clamped_ratio = max(0.0, min(progress_ratio, 2.0))
        progress_text = f"Fayda/Yatırım Oranı: {progress_ratio:.1%}"
    else:
        clamped_ratio = 0.0
        progress_text = "Fayda/Yatırım Oranı hesaplanamıyor"
    st.sidebar.progress(clamped_ratio / 2, text=progress_text)

fig_main = create_advanced_charts(df, summary, I, N, currency_symbol)
fig_comp = create_comparison_chart(summary, I, currency_symbol)

display_period = f"{analysis_years_value} yıl" if analysis_years_value > 1 else "1 yıl"

st.markdown('<h1 class="main-title">🚀 Otomasyon ROI & NPV Aracı</h1>', unsafe_allow_html=True)

if total_monthly_benefit <= 0:
    st.error(
        """
🚨 **Uyarı**: Hiçbir fayda kalemi aktif değil!

**Çözüm**: Yan menüden en az bir faydayı aktifleştirin:
- İşçilik tasarrufu (işçi sayısı > 0)
- Hurda/hata azaltma (oranlar farklı olsun)
- Zaman tasarrufu (saat > 0)
- Gelir artışı (baz gelir ve artış % > 0)
"""
    )
elif summary.get("NPV", 0) < 0:
    st.warning(
        """
⚠️ **Dikkat**: Proje ekonomik olarak uygun görünmüyor.

**Öneriler**:
- Yatırım tutarını azaltın
- Faydaları artırın
- İskonto oranını düşürün
"""
    )

st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
st.markdown(
    f'<h2 style="text-align: center; margin-bottom: 2rem; color: #1f77b4;">📊 Yatırım Analizi Özeti ({display_period})</h2>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    payback_value = summary.get("Payback (ay)")
    payback_text, payback_note = format_safe_metric(payback_value, "months", currency_symbol)
    if payback_value is None:
        payback_color = "#d62728"
        payback_icon = "🔴"
        payback_note = "Yatırım geri dönmüyor"
    else:
        payback_icon, payback_color = get_metric_color_and_icon("payback", payback_value)
        if not payback_note:
            payback_note = f"{format_currency(I, currency_symbol)} yatırımın amortisman süresi"

    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: {payback_color};">
        <div class="metric-title">{payback_icon} Yatırımın Geri Dönüş Süresi</div>
        <div class="metric-value" style="color: {payback_color};">{payback_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            {payback_note}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    total_return = summary.get("Toplam Aylık Fayda", 0.0)
    _, return_color = get_metric_color_and_icon("general", total_return)
    total_text, _ = format_safe_metric(total_return, "currency", currency_symbol)

    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: {return_color};">
        <div class="metric-title">💰 Yatırımın Toplam Getirisi ({display_period})</div>
        <div class="metric-value" style="color: {return_color};">{total_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            İndirimsiz brüt toplam fayda
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    alternative_return = summary.get("FV Alternatif", 0.0)
    alt_text, _ = format_safe_metric(alternative_return, "currency", currency_symbol)

    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: #ff7f0e;">
        <div class="metric-title">🏦 Yatırımın Alternatif Getirisi ({display_period})</div>
        <div class="metric-value" style="color: #ff7f0e;">{alt_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            %{r_annual*100:.1f} yıllık faizle alternatif yatırım
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    difference = summary.get("Toplam Aylık Fayda", 0.0) - summary.get("FV Alternatif", 0.0)
    diff_icon, diff_color = get_metric_color_and_icon("difference", difference)
    difference_text, _ = format_safe_metric(difference, "currency", currency_symbol)

    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: {diff_color};">
        <div class="metric-title">{diff_icon} Fark (Proje - Alternatif)</div>
        <div class="metric-value" style="color: {diff_color};">{difference_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            {"Proje daha karlı" if difference > 0 else "Alternatif daha karlı" if difference < 0 else "Getiriler eşit"}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col5:
    roi_value = summary.get("ROI (N ay) %", 0.0)
    roi_icon, roi_color = get_metric_color_and_icon("roi", roi_value)
    roi_text, _ = format_safe_metric(roi_value, "percentage", currency_symbol)

    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: {roi_color};">
        <div class="metric-title">{roi_icon} ROI Değeri ({display_period})</div>
        <div class="metric-value" style="color: {roi_color};">{roi_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            Yatırım getiri oranı
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="dashboard-container" style="margin-top: 1rem;">', unsafe_allow_html=True)

col_left_metrics, col_center_metrics, col_right_metrics = st.columns(3)

with col_left_metrics:
    npv_value = summary.get("NPV", 0.0)
    npv_icon, npv_color = get_metric_color_and_icon("npv", npv_value)
    npv_text, _ = format_safe_metric(npv_value, "currency", currency_symbol)
    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: {npv_color};">
        <div class="metric-title">{npv_icon} Net Bugünkü Değer (NPV)</div>
        <div class="metric-value" style="color: {npv_color};">{npv_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            %{r_annual*100:.1f} iskonto oranıyla bugünkü değer
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_center_metrics:
    breakeven_months = summary.get("Payback (ay)", None)
    if breakeven_months is not None:
        breakeven_amount = I
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: #9467bd;">
            <div class="metric-title">⚖️ Başabaş Analizi</div>
            <div class="metric-value" style="color: #9467bd;">{format_currency(breakeven_amount, currency_symbol)}</div>
            <div style="color: gray; font-size: 0.8rem;">
                {breakeven_months}. ayda başabaş noktası
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="metric-card" style="border-left-color: #d62728;">
            <div class="metric-title">⚖️ Başabaş Analizi</div>
            <div class="metric-value" style="color: #d62728;">Yok</div>
            <div style="color: gray; font-size: 0.8rem;">
                {N} ay içinde başabaş sağlanamıyor
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

with col_right_metrics:
    monthly_avg = summary.get("Toplam Aylık Fayda", 0.0) / N if N else 0
    monthly_avg_text, _ = format_safe_metric(monthly_avg, "currency", currency_symbol)
    st.markdown(
        f"""
    <div class="metric-card" style="border-left-color: #17becf;">
        <div class="metric-title">📅 Ortalama Aylık Getiri</div>
        <div class="metric-value" style="color: #17becf;">{monthly_avg_text}</div>
        <div style="color: gray; font-size: 0.8rem;">
            {N} ay ortalaması
        </div>
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

    format_dict = {
        "Aylık Fayda": lambda x: f"{x:,.0f} {currency_symbol}",
        "Nakit Akışı (CF)": lambda x: f"{x:,.0f} {currency_symbol}",
        "Kümülatif Net": lambda x: f"{x:,.0f} {currency_symbol}",
        "İşçilik Tasarrufu": lambda x: f"{x:,.0f} {currency_symbol}",
        "Hurda/Hata Tasarrufu": lambda x: f"{x:,.0f} {currency_symbol}",
        "Zaman Tasarrufu": lambda x: f"{x:,.0f} {currency_symbol}",
        "Gelir Artışı": lambda x: f"{x:,.0f} {currency_symbol}",
    }

    styled_df = df.style.format(format_dict).applymap(
        color_negative_red, subset=["Kümülatif Net", "Nakit Akışı (CF)"]
    )

    st.dataframe(styled_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚖️ Yatırım Karşılaştırması")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("📈 Özet Analiz")

    npv_value = summary.get("NPV", 0.0)
    npv_text, _ = format_safe_metric(npv_value, "currency", currency_symbol)
    if isinstance(npv_value, (float, int, np.floating, np.integer)) and npv_value > 0:
        st.success(f"✅ Proje ekonomik olarak uygun! NPV: {npv_text}")
    else:
        st.error(f"❌ Proje ekonomik olarak uygun değil. NPV: {npv_text}")

    payback_value = summary.get("Payback (ay)")
    payback_text, _ = format_safe_metric(payback_value, "months", currency_symbol)
    if isinstance(payback_value, (float, int, np.floating, np.integer)) and np.isfinite(payback_value):
        if payback_value <= 24:
            st.success(f"⚡ Hızlı geri dönüş! {payback_text}")
        elif payback_value <= 36:
            st.warning(f"⏳ Orta vadeli geri dönüş: {payback_text}")
        else:
            st.error(f"🐌 Uzun vadeli geri dönüş: {payback_text}")
    else:
        st.error("🐌 Uzun vadeli veya geri dönüş yok")

    if "amort_toggle" in locals() and amort_toggle:
        st.info(
            f"Muhasebe Görünümü: Aylık amortisman ~ {format_currency(I/float(amort_months), currency_symbol)}. (NPV'ye dahil edilmez)"
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"""
### 📋 Açıklamalar

**🔍 Metrik Açıklamaları:**
- **Geri Dönüş Süresi**: Yatırımın kendini amorti etme süresi
- **Toplam Getiri**: {display_period} boyunca toplam brüt fayda
- **Alternatif Getiri**: Aynı tutarın %{r_annual*100:.1f} faizle {display_period} değeri  
- **Fark**: Proje ile alternatif yatırım arasındaki getiri farkı
- **ROI**: Yatırım getiri oranı (ROI = (Toplam Fayda - Yatırım) / Yatırım × 100)
- **NPV**: %{r_annual*100:.1f} iskonto oranıyla bugünkü net değer

**💰 Para Birimi**: Tüm hesaplamalar {currency_name} ({currency_symbol}) cinsindendir.

**⏱️ Analiz Dönemi**: {display_period} ({N} ay) üzerinden hesaplanmıştır.
""",
    unsafe_allow_html=True,
)
