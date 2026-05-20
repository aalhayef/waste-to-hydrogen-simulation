"""
============================================================
EN8913 - Interactive HYSYS-Style Waste-to-Hydrogen Simulator
Bahrain Polytechnic - Faculty of Engineering
============================================================

Run with:    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="HYSYS-Style W2H Simulator | EN8913",
    page_icon="H2",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { padding-top: 0.4rem; padding-bottom: 0; }
  .block-container { padding-top: 0.6rem; padding-bottom: 2rem;
                     max-width: 100%; }
  h1, h2, h3 { color: #1E3A5F; }

  /* ====== ALWAYS-VISIBLE SIDEBAR ====== */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D1B2A 0%,#1E3A5F 100%);
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
    transform: none !important;
    visibility: visible !important;
    border-right: 4px solid #E07A1F;
  }
  section[data-testid="stSidebar"] > div { padding-top: 0.4rem; }

  button[kind="header"], [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"], button[aria-label="Close sidebar"],
  button[aria-label="Open sidebar"] {
    display: none !important;
  }

  section[data-testid="stSidebar"] * { color: #E8EEF4 !important; }
  section[data-testid="stSidebar"] h3 {
    color: #FFB74D !important;
    border-bottom: 1px solid #4A6485;
    padding-bottom: 0.25rem;
    margin-top: 0.7rem;
    font-size: 12.5px;
    letter-spacing: 1.5px;
  }

  /* Number-input styling so it looks like a data-sheet cell */
  section[data-testid="stSidebar"] div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid #4A6485 !important;
    border-radius: 3px !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input {
    color: #1E3A5F !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    text-align: right !important;
  }
  section[data-testid="stSidebar"] label {
    color: #CFD8E3 !important; font-weight: 600 !important;
    font-size: 11.5px !important;
  }
  section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #E07A1F !important; color: white !important;
    border: none !important;
  }

  /* Read-only "computed" field styling */
  .computed-row {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(255,255,255,0.06);
    border-left: 3px solid #FFB74D;
    padding: 6px 10px; margin: 4px 0; border-radius: 3px;
  }
  .computed-row .lbl { color:#CFD8E3; font-size:11.5px; font-weight:600; }
  .computed-row .val { color:#FFB74D; font-size:13.5px; font-weight:700; }

  .section-banner {
    background: #1E3A5F; color: white; padding: 0.4rem 0.8rem;
    border-radius: 4px; font-weight: 700; font-size: 12px;
    letter-spacing: 1.5px; margin-bottom: 0.4rem;
    border-left: 5px solid #E07A1F;
  }

  div[data-testid="stMetric"] {
    background: #F4F6F8; padding: 0.8rem; border-radius: 6px;
    border-left: 4px solid #1E5F8C;
  }
  div[data-testid="stMetricLabel"] { color: #5A6B7B; font-weight: 700; }
  div[data-testid="stMetricValue"] { color: #1E3A5F; }

  /* Equipment data-sheet button row */
  .stButton > button {
    background: #FFFFFF; color: #1E3A5F; border: 1.5px solid #1E3A5F;
    font-weight: 700; padding: 8px 12px;
  }
  .stButton > button:hover {
    background: #1E3A5F; color: white; border-color: #E07A1F;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header bar
# ============================================================
st.markdown(
    """
    <div style="background: linear-gradient(90deg,#0D1B2A 0%,#1E3A5F 100%);
                padding: 0.8rem 1.4rem; border-radius: 6px;
                border-left: 6px solid #E07A1F; margin-bottom: 0.6rem;
                display: flex; justify-content: space-between; align-items: center;">
      <div>
        <div style="color:#E07A1F;font-size:11px;letter-spacing:2px;
                    font-weight:700;">EN8913 &middot; PROCESS SIMULATION</div>
        <div style="color:white;font-size:22px;font-weight:700;
                    margin-top:2px;">Waste-to-Hydrogen Process Simulator</div>
      </div>
      <div style="text-align:right;">
        <div style="color:#90CAF9;font-size:11px;font-weight:600;">CASE</div>
        <div style="color:white;font-size:14px;">W2H_R101_Bahrain_v1</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Thermodynamic model
# ============================================================
R_GAS  = 8.314
T_REF  = 800.0
DH_RXN = 49_400.0
K_REF  = 12.0


def equilibrium_constant(T_K):
    return K_REF * np.exp(-DH_RXN / R_GAS * (1.0/T_K - 1.0/T_REF))


def h2_yield_equilibrium(T_C, sc_ratio, P_bar, eta_cat=0.90):
    T_K = T_C + 273.15
    K = equilibrium_constant(T_K)
    xi_grid = np.linspace(1e-4, min(0.999, sc_ratio - 1e-4), 600)
    n_tot = 1.0 + sc_ratio + 2*xi_grid
    y_CH3OH = (1 - xi_grid) / n_tot
    y_H2O   = (sc_ratio - xi_grid) / n_tot
    y_CO2   = xi_grid / n_tot
    y_H2    = 3*xi_grid / n_tot
    Kp_calc = (y_CO2 * y_H2**3 / (y_CH3OH * y_H2O)) * (P_bar**2)
    idx = np.argmin(np.abs(np.log(Kp_calc + 1e-30) - np.log(K + 1e-30)))
    xi_eq = xi_grid[idx]
    return xi_eq * eta_cat


# ============================================================
# Saturated-steam table  (Antoine equation for water, 99-374 °C)
#   log10(P_mmHg) = A - B / (T_C + C)
# ============================================================
ANT_A, ANT_B, ANT_C = 8.14019, 1810.94, 244.485

def saturation_pressure_bar(T_C):
    """Saturation pressure of water vapor [bar] at temperature T [°C].
    Valid 99 - 300 °C (Antoine equation, NIST)."""
    log10_P_mmHg = ANT_A - ANT_B / (T_C + ANT_C)
    P_mmHg = 10 ** log10_P_mmHg
    return P_mmHg / 750.062     # mmHg -> bar


# ============================================================
# Cached ANN
# ============================================================
@st.cache_resource
def train_ann():
    rng = np.random.default_rng(42)
    N = 1500
    T_s   = rng.uniform(500, 900, N)
    SC_s  = rng.uniform(1.0, 5.0, N)
    P_s   = rng.uniform(1.0, 10.0, N)
    COD_s = rng.uniform(1.0, 20.0, N)
    y = np.array([3.0 * h2_yield_equilibrium(T, sc, P)
                  for T, sc, P in zip(T_s, SC_s, P_s)])
    y = y * (0.5 + 0.5*(COD_s/20.0)) * rng.normal(1.0, 0.03, N)
    X = np.column_stack([T_s, SC_s, P_s, COD_s])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    ann = MLPRegressor(hidden_layer_sizes=(16, 16, 8), activation="relu",
                       solver="adam", learning_rate_init=0.005,
                       max_iter=2000, random_state=42, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=40).fit(
                       scaler.transform(Xtr), ytr)
    r2 = r2_score(yte, ann.predict(scaler.transform(Xte)))
    return ann, scaler, r2


with st.spinner("Loading ANN model (one-time)..."):
    ann_model, ann_scaler, ann_r2 = train_ann()


# ============================================================
# CONTROL PANEL  (left sidebar, always visible, NUMERIC INPUTS)
# ============================================================
st.sidebar.markdown(
    """
    <div style="background:#0D1B2A;padding:0.6rem 0.8rem;border-radius:6px;
                border-left:4px solid #E07A1F;margin-bottom:0.6rem;">
      <div style="color:#E07A1F;font-size:11px;letter-spacing:2px;
                  font-weight:700;">CONTROL PANEL</div>
      <div style="color:white;font-size:14px;font-weight:700;
                  margin-top:2px;">Operating Conditions</div>
      <div style="color:#90CAF9;font-size:10px;margin-top:3px;">
        Numeric data-sheet form &middot; type values directly</div>
    </div>
    """, unsafe_allow_html=True)


# ---- 1. WASTEWATER FEED ----
st.sidebar.markdown("### 1. WASTEWATER  FEED  (S-1)")
c1, c2 = st.sidebar.columns(2)
feed_kg_h = c1.number_input("Feed Rate [kg/h]",
                             min_value=0.5, max_value=200.0,
                             value=5.0, step=0.5, key="feed",
                             format="%.2f")
COD = c2.number_input("COD [g/L]",
                       min_value=0.5, max_value=50.0,
                       value=10.0, step=0.5, key="COD",
                       format="%.2f")
T_S1 = st.sidebar.number_input("Feed Temperature [°C]",
                                min_value=10.0, max_value=80.0,
                                value=25.0, step=1.0, key="T_S1",
                                format="%.1f")


# ---- 2. SATURATED STEAM ----
st.sidebar.markdown("### 2. SATURATED  STEAM  (S-2)")
T_steam = st.sidebar.number_input("Steam Temperature [°C]",
                                   min_value=100.0, max_value=300.0,
                                   value=180.0, step=5.0, key="T_steam",
                                   format="%.1f",
                                   help="Pressure auto-computed from "
                                        "saturated-steam table (Antoine eq.)")
P_steam = saturation_pressure_bar(T_steam)
st.sidebar.markdown(
    f"""
    <div class="computed-row">
      <span class="lbl">P_sat (computed) [bar]</span>
      <span class="val">{P_steam:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
sc = st.sidebar.number_input("Steam-to-Carbon Ratio [mol/mol]",
                              min_value=1.0, max_value=10.0,
                              value=3.0, step=0.1, key="sc",
                              format="%.2f")


# ---- 3. REACTOR ----
st.sidebar.markdown("### 3. REACTOR  R-101")
c3, c4 = st.sidebar.columns(2)
T_C = c3.number_input("T [°C]",
                       min_value=400.0, max_value=900.0,
                       value=750.0, step=10.0, key="T",
                       format="%.0f")
P   = c4.number_input("P [bar]",
                       min_value=1.0, max_value=20.0,
                       value=1.0, step=0.5, key="P",
                       format="%.2f")
eta = st.sidebar.number_input("Catalyst Activity  η",
                               min_value=0.30, max_value=1.00,
                               value=0.90, step=0.01, key="eta",
                               format="%.2f")


# ---- 4. UTILITY ----
st.sidebar.markdown("### 4. UTILITY  (E-101 / SOLAR-101)")
c5, c6 = st.sidebar.columns(2)
GHI = c5.number_input("Solar GHI [kWh/m²/d]",
                       min_value=2.0, max_value=10.0,
                       value=6.2, step=0.1, key="GHI",
                       format="%.2f")
collector_eff = c6.number_input("Collector η",
                                 min_value=0.20, max_value=0.85,
                                 value=0.55, step=0.01, key="ce",
                                 format="%.2f")


# ---- DISPLAY ----
st.sidebar.markdown("### DISPLAY")
animation_speed = st.sidebar.select_slider(
    "Flow Animation Speed",
    options=["Off", "Slow", "Normal", "Fast"], value="Normal",
    key="anim")
SPEED_MAP = {"Off": "0s", "Slow": "3s", "Normal": "1.5s", "Fast": "0.6s"}
anim_dur = SPEED_MAP[animation_speed]

st.sidebar.markdown(
    """
    <div style="margin-top:1rem;padding:0.6rem;background:rgba(255,255,255,0.06);
                border-radius:4px;font-size:10px;color:#90CAF9;line-height:1.5;">
       <b>Tip:</b> click any equipment-tag button below the PFD to open
       its <b>Process Data Sheet</b>.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Live calculations  (mass balance, kg/h)
# ============================================================
M_CH3OH = 0.03204
M_H2O   = 0.01802
M_H2    = 0.002016
M_CO2   = 0.04401

xi = h2_yield_equilibrium(T_C, sc, P, eta_cat=eta)
xi *= (0.5 + 0.5*(COD/20.0))

mol_feed = feed_kg_h / M_CH3OH
m_S1_org = feed_kg_h
mol_steam = sc * mol_feed
m_S2 = mol_steam * M_H2O
m_S3 = m_S1_org + m_S2
m_S4 = m_S3

mol_H2_out    = 3 * xi * mol_feed
mol_CO2_out   = xi * mol_feed
mol_unconv    = (1 - xi) * mol_feed
mol_H2O_unrxn = (sc - xi) * mol_feed

m_H2  = mol_H2_out  * M_H2
m_CO2 = mol_CO2_out * M_CO2
m_H2O_out = mol_H2O_unrxn * M_H2O
m_unconv  = mol_unconv * M_CH3OH

m_S5 = m_H2 + m_CO2 + m_H2O_out + m_unconv
m_P1 = m_H2
m_P2 = m_CO2
m_P3 = m_H2O_out + m_unconv

y_yield = 3 * xi
y_ann_pred = float(ann_model.predict(
    ann_scaler.transform([[T_C, sc, P, COD]]))[0])

DH_kJ_h    = (DH_RXN/1000) * mol_feed
Q_sens_kJ  = m_S3 * 2.2 * (T_C - 25)
Q_total_kW = (DH_kJ_h + Q_sens_kJ) / 3600

solar_kW_per_m2 = GHI / 8.0 * collector_eff
collector_area  = Q_total_kW / solar_kW_per_m2

CO2_smr            = 10.0
CO2_thiswork_solar = max(0.5, 1.1 * (1.0 - (eta - 0.9)*0.5))
CO2_reduction_pct  = (1 - CO2_thiswork_solar/CO2_smr) * 100

converged = xi > 0.05
status_color = "#2E8B57" if converged else "#C62828"
status_label = "CONVERGED" if converged else "NOT CONVERGED"


# ============================================================
# KPI Cards
# ============================================================
st.markdown(
    '<div class="section-banner">LIVE OUTPUTS</div>',
    unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("H₂ Yield (Thermo)", f"{y_yield:.2f} mol/mol")
k2.metric("H₂ Yield (ANN)", f"{y_ann_pred:.2f} mol/mol",
          delta=f"{(y_ann_pred - y_yield):+.2f}", delta_color="off")
k3.metric("H₂ Production", f"{m_H2:.2f} kg/h")
k4.metric("Reactor Heat Duty", f"{Q_total_kW:.2f} kW")
k5.metric("CO₂ vs SMR", f"−{CO2_reduction_pct:.0f}%")


# ============================================================
# Process Flow Diagram (animated SVG)
# ============================================================
st.markdown(
    '<div class="section-banner">PROCESS FLOW DIAGRAM &nbsp;&middot;&nbsp; '
    'PFD-101</div>',
    unsafe_allow_html=True)


def stream_line(x1, y1, x2, y2, color, label, label_offset=(0, -12),
                dash="10,8", dur=anim_dur):
    if dur == "0s":
        anim = ""
    else:
        anim = f'<animate attributeName="stroke-dashoffset" '\
               f'from="18" to="0" dur="{dur}" repeatCount="indefinite"/>'
    arrow_size = 8
    if x2 > x1:
        ax, ay = x2, y2
        triangle = f"M {ax-arrow_size},{ay-arrow_size/2} L {ax},{ay} "\
                   f"L {ax-arrow_size},{ay+arrow_size/2} Z"
    elif x2 < x1:
        ax, ay = x2, y2
        triangle = f"M {ax+arrow_size},{ay-arrow_size/2} L {ax},{ay} "\
                   f"L {ax+arrow_size},{ay+arrow_size/2} Z"
    elif y2 > y1:
        ax, ay = x2, y2
        triangle = f"M {ax-arrow_size/2},{ay-arrow_size} L {ax},{ay} "\
                   f"L {ax+arrow_size/2},{ay-arrow_size} Z"
    else:
        ax, ay = x2, y2
        triangle = f"M {ax-arrow_size/2},{ay+arrow_size} L {ax},{ay} "\
                   f"L {ax+arrow_size/2},{ay+arrow_size} Z"
    lx, ly = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
    return f"""
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
              stroke="{color}" stroke-width="3"
              stroke-dasharray="{dash}" stroke-linecap="round">
          {anim}
        </line>
        <path d="{triangle}" fill="{color}"/>
        <text x="{lx}" y="{ly}" font-size="11" font-weight="700"
              fill="{color}" text-anchor="middle"
              font-family="Calibri,Arial,sans-serif">{label}</text>
    """


def equipment_box(x, y, w, h, label, sublabel="", color="#FFFFFF",
                  border="#1E3A5F", text_color="#1E3A5F"):
    return f"""
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6"
            fill="{color}" stroke="{border}" stroke-width="2.5"
            filter="url(#shadow)"/>
      <text x="{x+w/2}" y="{y+h/2 - 2}" font-size="14" font-weight="700"
            fill="{text_color}" text-anchor="middle" dominant-baseline="middle"
            font-family="Calibri,Arial,sans-serif">{label}</text>
      <text x="{x+w/2}" y="{y+h/2 + 14}" font-size="10"
            fill="#5A6B7B" text-anchor="middle" dominant-baseline="middle"
            font-family="Calibri,Arial,sans-serif">{sublabel}</text>
    """


def value_chip(x, y, label, value, color="#1E5F8C"):
    w = max(85, 8 * len(value) + 30)
    return f"""
      <rect x="{x}" y="{y}" width="{w}" height="34" rx="4" ry="4"
            fill="white" stroke="{color}" stroke-width="1.2"/>
      <rect x="{x}" y="{y}" width="4" height="34" fill="{color}"/>
      <text x="{x+10}" y="{y+13}" font-size="9" font-weight="700"
            fill="#5A6B7B" font-family="Calibri,Arial,sans-serif">{label}</text>
      <text x="{x+10}" y="{y+27}" font-size="12" font-weight="700"
            fill="{color}" font-family="Calibri,Arial,sans-serif">{value}</text>
    """


C_FEED   = "#1976D2"
C_STEAM  = "#E53935"
C_MIXED  = "#7B1FA2"
C_HOT    = "#FB8C00"
C_RXN    = "#388E3C"
C_H2     = "#03A9F4"
C_CO2    = "#616161"
C_WATER  = "#0288D1"
C_HEAT   = "#FBC02D"

svg_w, svg_h = 1280, 480
svg = f"""
<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;background:#F4F6F8;border:1px solid #DDE3EA;
            border-radius:8px;">
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.20"/>
    </filter>
    <pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6"
             patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#FBC02D" stroke-width="2"/>
    </pattern>
  </defs>

  <rect x="0" y="0" width="{svg_w}" height="32" fill="#0D1B2A"/>
  <text x="20" y="21" font-size="13" font-weight="700" fill="#E07A1F"
        font-family="Calibri,Arial,sans-serif">PFD-101</text>
  <text x="80" y="21" font-size="13" font-weight="600" fill="white"
        font-family="Calibri,Arial,sans-serif">
    Wastewater Catalytic Steam Reformer  &#8211;  Bahrain Polytechnic
  </text>
  <rect x="{svg_w-150}" y="6" width="135" height="20" rx="3" ry="3"
        fill="{status_color}"/>
  <text x="{svg_w-82.5}" y="20" font-size="11" font-weight="700"
        fill="white" text-anchor="middle"
        font-family="Calibri,Arial,sans-serif">{status_label}</text>

  {stream_line(40, 165, 175, 165, C_FEED,
               f"S-1  Wastewater  {m_S1_org:.1f} kg/h", (0, -8))}
  {stream_line(40, 285, 175, 285, C_STEAM,
               f"S-2  Steam  {m_S2:.1f} kg/h", (0, 18))}

  <line x1="175" y1="165" x2="220" y2="225" stroke="{C_FEED}" stroke-width="3"/>
  <line x1="175" y1="285" x2="220" y2="225" stroke="{C_STEAM}" stroke-width="3"/>

  {stream_line(245, 225, 340, 225, C_MIXED,
               f"S-3  Mixed Feed  {m_S3:.1f} kg/h", (0, -8))}
  {stream_line(465, 225, 560, 225, C_HOT,
               f"S-4  Preheated  {T_C:.0f} °C", (0, -8))}
  {stream_line(720, 225, 815, 225, C_RXN,
               f"S-5  Reactor Outlet  {m_S5:.1f} kg/h", (0, -8))}
  {stream_line(945, 165, 1230, 165, C_H2,
               f"P-1  Hydrogen  {m_H2:.2f} kg/h", (0, -8))}
  {stream_line(945, 235, 1230, 235, C_CO2,
               f"P-2  CO2 Vent  {m_CO2:.2f} kg/h", (0, 18))}
  {stream_line(945, 305, 1230, 305, C_WATER,
               f"P-3  Condensate  {m_P3:.1f} kg/h", (0, 18))}
  {stream_line(405, 405, 405, 290, C_HEAT,
               f"Q  {Q_total_kW:.1f} kW", (45, -2))}

  <circle cx="232" cy="225" r="18" fill="white" stroke="#1E3A5F" stroke-width="2.5"
          filter="url(#shadow)"/>
  <text x="232" y="230" font-size="14" font-weight="700" fill="#1E3A5F"
        text-anchor="middle" font-family="Calibri,Arial,sans-serif">M</text>
  <text x="232" y="262" font-size="10" font-weight="700" fill="#1E3A5F"
        text-anchor="middle" font-family="Calibri,Arial,sans-serif">M-101</text>

  {equipment_box(340, 195, 125, 60, "E-101", "Pre-Heater",
                 color="#FFF3E0", border="#FB8C00", text_color="#E65100")}

  {equipment_box(560, 130, 160, 190, "R-101", "Catalytic Steam Reformer",
                 color="#E8F5E9", border="#388E3C", text_color="#1B5E20")}
  <rect x="575" y="160" width="130" height="140" fill="url(#hatch)" opacity="0.4"/>
  <text x="640" y="185" font-size="10" font-weight="600" fill="#1B5E20"
        text-anchor="middle" font-family="Calibri,Arial,sans-serif">Ni Catalyst Bed</text>

  {equipment_box(815, 110, 130, 230, "V-101", "Three-Phase Separator",
                 color="#E3F2FD", border="#1976D2", text_color="#0D47A1")}
  <line x1="822" y1="180" x2="938" y2="180" stroke="#1976D2"
        stroke-width="0.8" stroke-dasharray="3,3"/>
  <line x1="822" y1="260" x2="938" y2="260" stroke="#1976D2"
        stroke-width="0.8" stroke-dasharray="3,3"/>
  <text x="930" y="173" font-size="9" fill="#0D47A1" text-anchor="end"
        font-family="Calibri,Arial,sans-serif">Gas</text>
  <text x="930" y="220" font-size="9" fill="#0D47A1" text-anchor="end"
        font-family="Calibri,Arial,sans-serif">CO2</text>
  <text x="930" y="298" font-size="9" fill="#0D47A1" text-anchor="end"
        font-family="Calibri,Arial,sans-serif">Liquid</text>

  <rect x="345" y="395" width="120" height="50" rx="4" ry="4"
        fill="#FFF8E1" stroke="#FBC02D" stroke-width="2.5"
        filter="url(#shadow)"/>
  <text x="405" y="420" font-size="12" font-weight="700" fill="#F57F17"
        text-anchor="middle" font-family="Calibri,Arial,sans-serif">SOLAR-101</text>
  <text x="405" y="436" font-size="10" fill="#5A6B7B"
        text-anchor="middle" font-family="Calibri,Arial,sans-serif">
    {collector_area:.1f} m²
  </text>

  {value_chip(45, 105, "S-1 / T,P", f"{T_S1:.0f} °C, 1 bar", C_FEED)}
  {value_chip(45, 320, "S-2 / T,P", f"{T_steam:.0f} °C, {P_steam:.1f} bar", C_STEAM)}
  {value_chip(560, 80, "R-101 / T", f"{T_C:.0f} °C", C_RXN)}
  {value_chip(720, 80, "S/C", f"{sc:.1f}", C_RXN)}
  {value_chip(995, 105, "P-1 H2", f"{m_H2:.2f} kg/h", C_H2)}
</svg>
"""

components.html(
    f"""<!doctype html><html><head><meta charset="utf-8">
    <style>html,body{{margin:0;padding:0;background:transparent;
                       font-family:Calibri,Arial,sans-serif;}}</style>
    </head><body>{svg}</body></html>""",
    height=505, scrolling=False)


# ============================================================
# EQUIPMENT DATA SHEET POPUPS
# ============================================================
st.markdown(
    '<div class="section-banner">EQUIPMENT  PROCESS DATA SHEETS '
    '&nbsp;&middot;&nbsp; click a tag to view</div>',
    unsafe_allow_html=True)


def datasheet_table(rows):
    """Render a list of (key, value) rows as a clean HTML table."""
    cells = "".join(
        f"<tr><td style='padding:6px 10px;background:#F4F6F8;"
        f"border-bottom:1px solid #DDE3EA;width:42%;'>"
        f"<b style='color:#5A6B7B;font-size:11.5px;letter-spacing:0.4px;'>"
        f"{k.upper()}</b></td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #DDE3EA;"
        f"color:#1E3A5F;font-size:13px;'>{v}</td></tr>"
        for k, v in rows
    )
    return ("<table style='width:100%;border-collapse:collapse;"
            "border:1px solid #DDE3EA;border-radius:4px;overflow:hidden;'>"
            f"{cells}</table>")


def datasheet_header(tag, name, color):
    return f"""
    <div style="background:{color};padding:0.7rem 1rem;border-radius:6px;
                margin-bottom:0.6rem;display:flex;align-items:center;
                justify-content:space-between;">
      <div>
        <div style="color:rgba(255,255,255,0.85);font-size:11px;
                    letter-spacing:1.5px;font-weight:700;">PROCESS DATA SHEET</div>
        <div style="color:white;font-size:18px;font-weight:700;
                    margin-top:1px;">{tag} &middot; {name}</div>
      </div>
      <div style="background:rgba(255,255,255,0.18);padding:4px 10px;
                  border-radius:3px;color:white;font-size:11px;
                  font-weight:700;letter-spacing:1px;">EN8913 / W2H</div>
    </div>
    """


# ---- Data sheets ----
@st.dialog("Process Data Sheet — M-101", width="large")
def show_M101():
    st.markdown(datasheet_header("M-101", "Inline Static Mixer", "#1E3A5F"),
                unsafe_allow_html=True)
    st.markdown(datasheet_table([
        ("Equipment Type", "Static (motionless) inline mixer / T-junction"),
        ("Function / Role",
         "Combines wastewater feed (S-1) with saturated steam (S-2) "
         "into a homogeneous two-phase stream (S-3) before pre-heating."),
        ("Construction Material", "Stainless Steel SS-316 L"),
        ("Pressure Rating",       "ANSI Class 150"),
        ("Design Temperature",    f"{max(T_S1, T_steam) + 50:.0f} °C"),
        ("Pressure Drop",         "&lt; 0.05 bar (negligible)"),
        ("Inlet Streams",         "S-1 (wastewater) + S-2 (steam)"),
        ("Outlet Stream",         f"S-3 — {m_S3:.1f} kg/h"),
        ("Mixing Mechanism",      "Helical static elements, 6 stages"),
        ("Reference",             "Perry's Chemical Engineers' Handbook (2008)"),
    ]), unsafe_allow_html=True)


@st.dialog("Process Data Sheet — E-101", width="large")
def show_E101():
    st.markdown(datasheet_header("E-101", "Pre-Heater  (Solar-Driven)",
                                 "#FB8C00"), unsafe_allow_html=True)
    st.markdown(datasheet_table([
        ("Equipment Type",
         "Shell-and-tube heat exchanger (solar thermal-oil heated)"),
        ("Function / Role",
         "Heats the mixed feed (S-3) from mixing temperature up to the "
         "reactor inlet temperature (S-4)."),
        ("Configuration",         "1-2 shell &amp; tube, counter-current"),
        ("Heat Transfer Area",    f"≈ {Q_total_kW * 0.7:.1f} m² (sized for "
                                  "U = 350 W/m²·K, LMTD = 50 °C)"),
        ("Tube Material",         "Inconel 625 (high-T corrosion resistance)"),
        ("Heating Medium",        "Therminol VP-1 from SOLAR-101"),
        ("Heat Duty (current)",   f"<b>{Q_total_kW:.2f} kW</b>"),
        ("Inlet Stream (cold)",   f"S-3 @ ≈ 110 °C"),
        ("Outlet Stream (cold)",  f"S-4 @ {T_C:.0f} °C"),
        ("Design Pressure",       f"{P:.1f} bar"),
        ("Design Temperature",    "850 °C"),
        ("Reference",
         "Cengel &amp; Ghajar (2015) — <i>Heat &amp; Mass Transfer</i>"),
    ]), unsafe_allow_html=True)


@st.dialog("Process Data Sheet — R-101", width="large")
def show_R101():
    st.markdown(datasheet_header("R-101", "Catalytic Steam Reformer",
                                 "#388E3C"), unsafe_allow_html=True)
    st.markdown(datasheet_table([
        ("Equipment Type",
         "Vertical fixed-bed catalytic reactor"),
        ("Function / Role",
         "Converts methanol-equivalent organics in wastewater "
         "into a hydrogen-rich syngas via catalytic steam reforming "
         "(CH<sub>3</sub>OH + H<sub>2</sub>O &rarr; CO<sub>2</sub> + 3 H<sub>2</sub>)."),
        ("Catalyst",              "Ni / Al<sub>2</sub>O<sub>3</sub> "
                                  "(15 wt% Ni, 3 mm spheres)"),
        ("Catalyst Loading",      "≈ 70 kg"),
        ("Bed Height",            "1.50 m"),
        ("Bed Diameter",          "0.30 m"),
        ("GHSV (typical)",        "5 000 – 10 000 h⁻¹"),
        ("Operating Temperature", f"<b>{T_C:.0f} °C</b>  "
                                  "(range 500 – 900 °C)"),
        ("Operating Pressure",    f"<b>{P:.2f} bar</b>"),
        ("Catalyst Activity η",   f"<b>{eta:.2f}</b>"),
        ("Conversion (xi)",       f"{xi:.3f} mol/mol feed"),
        ("H<sub>2</sub> Yield",   f"{y_yield:.2f} mol H<sub>2</sub>/mol feed"),
        ("Heat of Reaction",      "+49.4 kJ/mol (endothermic)"),
        ("Material",              "Inconel 800 H (creep-resistant)"),
        ("Reference",
         "Rahman et al. (2021); Perry's Handbook (2008)"),
    ]), unsafe_allow_html=True)


@st.dialog("Process Data Sheet — V-101", width="large")
def show_V101():
    st.markdown(datasheet_header("V-101", "Three-Phase Separator",
                                 "#1976D2"), unsafe_allow_html=True)
    st.markdown(datasheet_table([
        ("Equipment Type",
         "Vertical knock-out drum with mesh demister"),
        ("Function / Role",
         "Separates the reactor outlet (S-5) into three product streams: "
         "H<sub>2</sub> product gas (P-1), CO<sub>2</sub> vent (P-2), "
         "and condensed unreacted water + organics (P-3)."),
        ("Internal Diameter",     "0.40 m"),
        ("Tangent-to-Tangent",    "1.20 m"),
        ("Liquid Residence Time", "5 min"),
        ("Operating Temperature", f"≈ {T_C-50:.0f} °C "
                                  "(after partial cool-down)"),
        ("Operating Pressure",    f"{P:.2f} bar"),
        ("Internals",             "Wire-mesh demister (150 mm thick), "
                                  "vortex breaker, inlet baffle"),
        ("Material",              "SS-316 L (CO<sub>2</sub>/H<sub>2</sub>O atmosphere)"),
        ("P-1 (H<sub>2</sub>) flow", f"{m_H2:.3f} kg/h"),
        ("P-2 (CO<sub>2</sub>) flow", f"{m_CO2:.3f} kg/h"),
        ("P-3 (Condensate) flow",  f"{m_P3:.2f} kg/h"),
        ("Reference",
         "McCabe, Smith &amp; Harriott (2005) — <i>Unit Operations</i>"),
    ]), unsafe_allow_html=True)


@st.dialog("Process Data Sheet — SOLAR-101", width="large")
def show_SOLAR101():
    st.markdown(datasheet_header("SOLAR-101", "Solar Thermal Collector",
                                 "#F57F17"), unsafe_allow_html=True)
    st.markdown(datasheet_table([
        ("Equipment Type",
         "Parabolic-trough solar collector (PTC)"),
        ("Function / Role",
         "Captures direct solar radiation and supplies the heat duty of "
         "E-101 via a closed thermal-oil loop. Replaces fossil-fuel firing "
         "to give a near-zero-carbon hydrogen pathway."),
        ("Aperture Area (sized)", f"<b>{collector_area:.1f} m²</b>"),
        ("Concentration Ratio",   "≈ 30 ×"),
        ("Heat Transfer Fluid",   "Therminol VP-1 (synthetic oil)"),
        ("HTF Outlet Temperature","400 °C"),
        ("Collector Efficiency η",f"<b>{collector_eff:.2f}</b>"),
        ("Solar Resource (GHI)",  f"{GHI:.2f} kWh/m²/day "
                                  "(Manama, Bahrain — 26 °N)"),
        ("Operating Hours",       "8 h/day (effective)"),
        ("Tracking",              "Single-axis E–W"),
        ("Heat Duty Delivered",   f"{Q_total_kW:.2f} kW"),
        ("Reference",             "IRENA (2022); ASHRAE Handbook (2017)"),
    ]), unsafe_allow_html=True)


# Equipment buttons row
b1, b2, b3, b4, b5 = st.columns(5)
if b1.button("📋 M-101  Mixer", use_container_width=True):
    show_M101()
if b2.button("📋 E-101  Pre-Heater", use_container_width=True):
    show_E101()
if b3.button("📋 R-101  Reactor", use_container_width=True):
    show_R101()
if b4.button("📋 V-101  Separator", use_container_width=True):
    show_V101()
if b5.button("📋 SOLAR-101  Collector", use_container_width=True):
    show_SOLAR101()


# ============================================================
# OUTCOME GRAPHS
# ============================================================
st.markdown(
    '<div class="section-banner">OUTCOME GRAPHS &nbsp;&middot;&nbsp; '
    'LIVE PARAMETRIC RESPONSE</div>',
    unsafe_allow_html=True)

plt.rcParams.update({
    "figure.dpi": 100, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

g_left, g_right = st.columns(2)

T_range = np.arange(500, 901, 20)
y_T = np.array([3.0 * h2_yield_equilibrium(t, sc, P, eta) *
                (0.5 + 0.5*(COD/20.0)) for t in T_range])
fig1, ax1 = plt.subplots(figsize=(6.2, 3.5))
ax1.plot(T_range, y_T, color="#1E5F8C", lw=2.5, label="Equilibrium curve")
ax1.scatter([T_C], [y_yield], color="#E07A1F", s=180, zorder=5,
            edgecolor="white", linewidth=2, label="Operating point")
ax1.axhline(3.0, ls="--", color="#5A6B7B", lw=1, alpha=0.7,
            label="Stoichiometric max")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("H₂ Yield (mol/mol)")
ax1.set_title(f"Effect of Temperature  (S/C={sc}, P={P:.1f} bar)")
ax1.legend(loc="lower right", fontsize=9)
fig1.tight_layout()
g_left.pyplot(fig1, use_container_width=True)
plt.close(fig1)

SC_range = np.arange(1.0, 5.01, 0.1)
y_SC = np.array([3.0 * h2_yield_equilibrium(T_C, s, P, eta) *
                 (0.5 + 0.5*(COD/20.0)) for s in SC_range])
fig2, ax2 = plt.subplots(figsize=(6.2, 3.5))
ax2.plot(SC_range, y_SC, color="#2E8B57", lw=2.5, label="Equilibrium curve")
ax2.scatter([sc], [y_yield], color="#E07A1F", s=180, zorder=5,
            edgecolor="white", linewidth=2, label="Operating point")
ax2.set_xlabel("Steam-to-Carbon Ratio (mol/mol)")
ax2.set_ylabel("H₂ Yield (mol/mol)")
ax2.set_title(f"Effect of S/C Ratio  (T={T_C:.0f} °C, P={P:.1f} bar)")
ax2.legend(loc="lower right", fontsize=9)
fig2.tight_layout()
g_right.pyplot(fig2, use_container_width=True)
plt.close(fig2)

g_left2, g_right2 = st.columns(2)

fig3, ax3 = plt.subplots(figsize=(6.2, 3.5))
labels = ["Reaction\nEnthalpy", "Sensible\nHeating"]
values = [DH_kJ_h/3600, Q_sens_kJ/3600]
bars = ax3.bar(labels, values, color=["#1E5F8C", "#E07A1F"], width=0.5)
ax3.set_ylabel("Heat Duty (kW)")
ax3.set_title(f"Reactor Heat Budget = {Q_total_kW:.2f} kW  →  "
              f"Solar Area = {collector_area:.1f} m²")
ax3.set_ylim(0, max(values, default=1)*1.30)
for b, v in zip(bars, values):
    ax3.text(b.get_x() + b.get_width()/2, v + max(values)*0.02,
             f"{v:.2f} kW", ha="center", fontweight="bold")
fig3.tight_layout()
g_left2.pyplot(fig3, use_container_width=True)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(6.2, 3.5))
pathways = ["SMR", "SMR\n+CCS", "Grid\nElectrolysis",
            "This Work\n(Wastewater)", "This Work\n(+ Solar)"]
co2 = [10.0, 4.5, 25.0, 3.2, CO2_thiswork_solar]
colors = ["#5A6B7B", "#5A6B7B", "#5A6B7B", "#1E5F8C", "#2E8B57"]
bars = ax4.bar(pathways, co2, color=colors, width=0.55)
for b, v in zip(bars, co2):
    ax4.text(b.get_x() + b.get_width()/2, v + 0.5,
             f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
ax4.set_ylabel("kg CO₂ per kg H₂")
ax4.set_title(f"Carbon Intensity  ({CO2_reduction_pct:.0f}% reduction vs SMR)")
ax4.set_ylim(0, max(co2)*1.18)
plt.setp(ax4.get_xticklabels(), fontsize=8)
fig4.tight_layout()
g_right2.pyplot(fig4, use_container_width=True)
plt.close(fig4)


# ============================================================
# Stream + Equipment summary tables
# ============================================================
st.markdown(
    '<div class="section-banner">STREAM &amp; EQUIPMENT SUMMARY</div>',
    unsafe_allow_html=True)

left, right = st.columns([3, 2])

with left:
    st.markdown("**Material Stream Summary**")
    streams = pd.DataFrame({
        "Stream":        ["S-1", "S-2", "S-3", "S-4", "S-5",
                          "P-1", "P-2", "P-3"],
        "Description":   ["Wastewater Feed", "Saturated Steam", "Mixed Feed",
                          "Preheated Feed", "Reactor Outlet",
                          "Hydrogen Product", "CO₂ Vent", "Condensate"],
        "T [°C]":        [T_S1, T_steam, (T_S1 + T_steam)/2,
                          T_C, T_C, T_C-50, T_C-50, 80],
        "P [bar]":       [1.0, round(P_steam, 2), P, P, P, P, P, P],
        "Mass [kg/h]":   [round(m_S1_org, 2), round(m_S2, 2),
                          round(m_S3, 2), round(m_S4, 2),
                          round(m_S5, 2), round(m_H2, 3),
                          round(m_CO2, 3), round(m_P3, 2)],
        "Phase":         ["L", "V", "L+V", "V", "V", "V", "V", "L"],
    })
    st.dataframe(streams, use_container_width=True, hide_index=True,
                 height=320)

with right:
    st.markdown("**Equipment Summary**")
    equipment = pd.DataFrame({
        "Tag":         ["M-101", "E-101", "R-101", "V-101", "SOLAR-101"],
        "Description": ["Mixer", "Pre-Heater",
                        "Steam Reformer (Ni cat.)",
                        "3-Phase Separator", "Solar Collector"],
        "Duty/Spec":   ["—",
                        f"{Q_total_kW:.1f} kW",
                        f"T={T_C:.0f}°C, η={eta:.2f}",
                        f"P={P:.1f} bar",
                        f"{collector_area:.1f} m²"],
    })
    st.dataframe(equipment, use_container_width=True, hide_index=True,
                 height=220)
    st.markdown(
        f"""
        <div style="background:#F4F6F8;padding:0.7rem;border-radius:6px;
                    border-left:4px solid #1E5F8C;margin-top:0.4rem;">
          <div style="font-size:10px;color:#5A6B7B;font-weight:700;
                      letter-spacing:1px;">ANN MODEL DIAGNOSTICS</div>
          <div style="font-size:13px;color:#1E3A5F;margin-top:4px;">
            Architecture: 16-16-8 ReLU&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            Test R² = <b>{ann_r2:.3f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown(
    """
    <div style="color:#5A6B7B;font-size:11px;text-align:center;
                margin-top:1rem;border-top:1px solid #DDE3EA;padding-top:0.5rem;">
       EN8913 - Waste-to-Hydrogen Production from Industrial Wastewater  |
       Bahrain Polytechnic - Faculty of Engineering  |
       Abdulaziz Hayef Alshammari
    </div>
    """, unsafe_allow_html=True)
