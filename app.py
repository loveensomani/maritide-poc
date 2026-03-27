"""
MariTide PharmaACE Model — Light Theme Dashboard
Clean light theme + blue accents. Executive sidebar levers + Detailed Controls tab.
All charts show base case as dotted reference line for comparison.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from diffusion import diffusion_l

# ─── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(page_title="MariTide PharmaACE", layout="wide", page_icon="💊",
                   initial_sidebar_state="expanded")

# ─── Theme Constants ──────────────────────────────────────────────────────

ACCENT = "#2563EB"
ACCENT_DIM = "#1D4ED8"
ACCENT_GLOW = "rgba(37,99,235,0.12)"
ACCENT2 = "#7C3AED"
ACCENT3 = "#059669"
PAGE_BG = "#F8FAFC"
PANEL_BG = "#FFFFFF"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E2E8F0"
MUTED = "#94A3B8"
LIGHT = "#1E293B"
WHITE = "#FFFFFF"

# Backwards-compat aliases
GOLD = ACCENT
GOLD_DIM = ACCENT_DIM
GOLD_GLOW = ACCENT_GLOW

DISEASE_COLORS = {
    'T2D': '#2563EB', 'HFpEF': '#7C3AED', 'ASCVD': '#DC2626',
    'CKD': '#059669', 'OSA': '#A855F7', 'MASH': '#EA580C',
    'No ORCS': '#0891B2', 'All Others': '#64748B',
}
WAVE_COLORS = {
    'MariTide': '#2563EB', 'W1': '#DC2626', 'W2': '#7C3AED', 'W3': '#A855F7',
}

FUNNEL_COLORS = ['#2563EB', '#059669', '#7C3AED', '#DC2626', '#0891B2']

# ─── CSS Injection ────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    .stApp {
        background: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #CBD5E1;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1E293B !important; font-size: 0.82rem; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 0.3rem; font-weight: 700;
    }
    section[data-testid="stSidebar"] .stSlider label { color: #64748B !important; font-size: 0.78rem; font-weight: 500; }
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] { color: #94A3B8 !important; }
    section[data-testid="stSidebar"] hr { border-color: #E2E8F0; }

    /* ── Metrics — clean cards, equal height ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.03);
        height: 108px;
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important; font-weight: 800; font-size: 1.6rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important; font-size: 0.7rem !important;
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
    }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; font-weight: 600; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 2px solid #E2E8F0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #94A3B8 !important; font-weight: 600; font-size: 0.85rem;
        padding: 0.7rem 1.6rem; border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover { color: #1E293B !important; }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="stExpander"] summary span { color: #1E293B !important; font-weight: 600; }
    div[data-testid="stExpander"]:hover { border-color: #CBD5E1; }

    /* ── Data editor / dataframe ── */
    .stDataFrame, .stDataEditor {
        border-radius: 10px; overflow: hidden;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── Card classes ── */
    .nexa-card {
        background: #FFFFFF; border: 1px solid #CBD5E1;
        border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .nexa-card h4 { color: #2563EB; margin: 0 0 0.6rem 0; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .nexa-label { color: #64748B; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.15rem; font-weight: 600; }
    .nexa-value { color: #0F172A; font-size: 1.3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .nexa-sub { color: #64748B; font-size: 0.78rem; }

    /* ── Sidebar change pills ── */
    .change-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        border-radius: 20px;
        padding: 0.22rem 0.75rem;
        font-size: 0.73rem; font-weight: 600;
        margin: 0.15rem 0.1rem;
    }
    .no-changes { color: #94A3B8; font-size: 0.78rem; font-style: italic; }

    /* ── Section headers ── */
    .section-header {
        color: #0F172A; font-size: 0.95rem; font-weight: 700;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem; margin: 1.2rem 0 0.8rem 0;
    }
    .section-header span { color: #2563EB; }

    /* ── Filter chip area ── */
    .filter-row {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem; margin-bottom: 0.8rem;
    }

    /* ── Status bar ── */
    .status-bar {
        background: #EFF6FF;
        border-left: 3px solid #2563EB;
        padding: 0.6rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
    }
    .status-bar-base {
        background: #F0FDF4;
        border-left: 3px solid #059669;
        padding: 0.6rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
    }

    /* ── Subheader override ── */
    .stApp h2 { color: #0F172A !important; font-size: 0.95rem !important; font-weight: 700 !important; }
    .stApp h3 { color: #0F172A !important; font-size: 0.88rem !important; font-weight: 600 !important; }

    /* ── Selectbox / multiselect ── */
    .stSelectbox label, .stMultiSelect label {
        color: #64748B !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
    }

    /* ── Dividers ── */
    hr { border-color: #E2E8F0 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #2563EB;
        border: 1px solid #2563EB;
        color: #FFFFFF;
        font-weight: 600; font-size: 0.8rem;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
        box-shadow: 0 2px 8px rgba(37,99,235,0.25);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

    /* ── Chart containers ── */
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Layout Helper ─────────────────────────────────────────────────

def nexa_layout(title="", height=400, yaxis_title="", yaxis_range=None,
                xrange=None, legend_below=True, barmode=None):
    layout = dict(
        height=height,
        margin=dict(t=35 if title else 12, b=48, l=55, r=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155', size=11, family="Inter, sans-serif"),
        xaxis=dict(gridcolor='#F1F5F9', gridwidth=1, zeroline=False,
                   range=xrange or [2027.5, 2050.5],
                   tickfont=dict(color='#94A3B8', size=10, family="JetBrains Mono, monospace")),
        yaxis=dict(title=dict(text=yaxis_title, font=dict(color='#64748B', size=10)),
                   gridcolor='#F1F5F9', gridwidth=1, zeroline=False,
                   tickfont=dict(color='#94A3B8', size=10, family="JetBrains Mono, monospace")),
        hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='#E2E8F0',
                        font=dict(color='#1E293B', size=11, family="Inter, sans-serif")),
        hovermode='x unified',
    )
    if title:
        layout['title'] = dict(text=title, font=dict(color=ACCENT, size=13, family="Inter, sans-serif"))
    if yaxis_range:
        layout['yaxis']['range'] = yaxis_range
    if legend_below:
        layout['legend'] = dict(orientation='h', y=-0.18, font=dict(size=10, color=MUTED),
                                bgcolor='rgba(0,0,0,0)')
    if barmode:
        layout['barmode'] = barmode
    return layout


# ─── Caching + Data Loading ──────────────────────────────────────────────

@st.cache_data
def load_params():
    base_dir = Path(__file__).parent
    pkl_path = base_dir / "params_v3.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def excel_date_to_dt(val):
    from datetime import timedelta
    if val is None: return None
    if isinstance(val, datetime): return val
    if isinstance(val, (int, float)):
        return datetime(1899, 12, 30) + timedelta(days=int(val))
    return val


def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, datetime): return default
    try: return float(val)
    except: return default


def read_row_values_xl(ws, row, col_start, col_end):
    return [safe_float(ws.cell(row=row, column=c).value) for c in range(col_start, col_end + 1)]


def read_diffusion_row_xl(ws, row):
    phases = []
    for phase_start in [7, 13, 19]:
        phases.append({
            'start': safe_float(ws.cell(row=row, column=phase_start).value),
            'shape': safe_float(ws.cell(row=row, column=phase_start+1).value),
            'peak': safe_float(ws.cell(row=row, column=phase_start+2).value),
            'duration': safe_float(ws.cell(row=row, column=phase_start+3).value),
            'launch': excel_date_to_dt(ws.cell(row=row, column=phase_start+4).value),
        })
    overrides = read_row_values_xl(ws, row, 25, 53)
    return phases, overrides


def read_input_sheet_xl(ws, row_start, row_end, dc_start, dc_end):
    rows = []
    for r in range(row_start, row_end + 1):
        labels = {}
        for c_idx, c_name in [(2,'B'),(3,'C'),(4,'D'),(5,'E'),(6,'F')]:
            val = ws.cell(row=r, column=c_idx).value
            if val is not None:
                labels[c_name] = str(val).strip()
        if not labels: continue
        key = labels.get('F','') or labels.get('E','') or labels.get('D','')
        vals = read_row_values_xl(ws, r, dc_start, dc_end)
        phases, overrides = read_diffusion_row_xl(ws, r)
        rows.append({'labels': labels, 'key': key, 'values': vals,
                     'phases': phases, 'overrides': overrides, 'row': r})
    return rows


@st.cache_data(show_spinner="Extracting parameters from Excel…")
def extract_params_from_excel(file_bytes):
    import io, openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, keep_vba=True)
    p = {}
    p['years'] = list(range(2022, 2051))
    p['n_years'] = 29
    p['diseases'] = ['T2D', 'HFpEF', 'ASCVD', 'OSA', 'CKD', 'MASH', 'No ORCS', 'All Others']
    p['flow_sheets'] = ['T2D', 'HFpEF', 'ASCVD', 'OSA', 'CKD', 'MASH', 'No ORCS', 'All Others']

    # Year integers
    ws = wb['Treated']
    year_ints = []
    for c in range(55, 84):
        v = ws.cell(row=3, column=c).value
        if isinstance(v, datetime): year_ints.append(v.year)
        elif isinstance(v, (int, float)): year_ints.append(int(v))
        else: year_ints.append(2022 + c - 55)
    p['year_ints'] = year_ints

    # EPI
    ws = wb['EPI']
    epi_rows = []
    for r in range(5, 277):
        labels = {}
        for c_idx, c_name in [(2,'B'),(3,'C'),(4,'D'),(5,'E'),(6,'F'),(7,'G')]:
            val = ws.cell(row=r, column=c_idx).value
            if val is not None: labels[c_name] = str(val).strip()
        if not labels: continue
        vals = read_row_values_xl(ws, r, 8, 36)
        key = labels.get('G','') or labels.get('F','')
        epi_rows.append({'labels': labels, 'key': key, 'values': vals, 'row': r})
    p['epi'] = epi_rows

    # Input sheets
    for sheet_name, key, r_start, r_end, dc_start, dc_end in [
        ('Treated','treated',6,207,55,83), ('Coverage','coverage',6,118,55,83),
        ('GLP-1','glp1',6,31,55,83), ('1L-2L split','l1l2_split',6,46,55,83),
        ('Inj SOB','inj_sob',6,83,55,83), ('Access','access',6,159,55,83),
        ('Supply-Fulfillment','supply',7,83,55,83), ('Compliance','compliance',6,51,55,83),
        ('Cash-Pay','cashpay',5,155,55,83),
    ]:
        ws = wb[sheet_name]
        p[key] = read_input_sheet_xl(ws, r_start, r_end, dc_start, dc_end)

    # MariTide Shares
    ws = wb['MariTide Shares']
    p['maritide_shares'] = read_input_sheet_xl(ws, 162, 311, 61, 89)
    p['maritide_shares_raw'] = read_input_sheet_xl(ws, 5, 158, 61, 89)
    for row_data in p['maritide_shares_raw'] + p['maritide_shares']:
        r = row_data['row']
        row_data['phases'].append({
            'start': safe_float(ws.cell(row=r, column=25).value),
            'shape': safe_float(ws.cell(row=r, column=26).value),
            'peak': safe_float(ws.cell(row=r, column=27).value),
            'duration': safe_float(ws.cell(row=r, column=28).value),
            'launch': excel_date_to_dt(ws.cell(row=r, column=29).value),
        })

    # Pricing
    ws = wb['Pricing']
    pricing = {}
    for r in range(3, 53):
        label = ws.cell(row=r, column=2).value
        if label:
            pricing[str(label).strip()] = read_row_values_xl(ws, r, 45, 73)
    p['pricing'] = pricing

    # Flow sheets
    for sheet_name in p['flow_sheets']:
        ws = wb[sheet_name]
        flow_data = {}
        for r in range(1, 710):
            row_labels = [str(ws.cell(row=r, column=c).value).strip()
                          for c in range(1, 7) if ws.cell(row=r, column=c).value is not None]
            if not row_labels: continue
            vals = read_row_values_xl(ws, r, 7, 35)
            flow_data[f"row_{r}"] = {'labels': row_labels, 'label_str': '|'.join(row_labels), 'values': vals}
        p[f'flow_{sheet_name}'] = flow_data

    wb.close()
    return p


@st.cache_data
def compute_base_case(_p):
    results, inputs = run_full_model(_p, overrides=None)
    return results, inputs


# ─── Model Engine (PRESERVED — do not modify) ────────────────────────────

def compute_from_diffusion(phases, overrides, year_dates, year_ints, n_years=29):
    result = np.zeros(n_years)
    p1 = phases[0]
    if p1['launch'] is None:
        for i in range(n_years):
            result[i] = overrides[i] if i < len(overrides) else 0.0
        return result
    p1_launch_year = p1['launch'].year if isinstance(p1['launch'], datetime) else 0
    for i in range(n_years):
        yr = year_ints[i]
        t = year_dates[i]
        if yr < p1_launch_year:
            val = 0.0
        else:
            val = p1['start'] + diffusion_l(
                p1['shape'], p1['peak'] - p1['start'],
                p1['duration'], p1['launch'], t, 1)
            for ph_idx in range(1, len(phases)):
                ph = phases[ph_idx]
                if ph['launch'] is not None and ph['duration'] != 0:
                    ph_yr = ph['launch'].year if isinstance(ph['launch'], datetime) else 0
                    if yr >= ph_yr:
                        val += diffusion_l(
                            ph['shape'], ph['peak'] - ph['start'],
                            ph['duration'], ph['launch'], t, 1)
        if i < len(overrides):
            val += overrides[i]
        result[i] = val
    return result


def recompute_input_sheet(input_data, year_dates, year_ints, n_years=29):
    recomputed = []
    for row in input_data:
        phases = row.get('phases', [])
        overrides = row.get('overrides', [0.0] * n_years)
        if not phases or all(ph['launch'] is None for ph in phases):
            recomputed.append({'key': row['key'], 'values': row['values'], 'recomputed': False})
        else:
            new_vals = compute_from_diffusion(phases, overrides, year_dates, year_ints, n_years)
            recomputed.append({'key': row['key'], 'values': list(new_vals), 'recomputed': True})
    return recomputed


def recompute_maritide_shares(p, year_dates, year_ints, n_years=29):
    raw_data = p['maritide_shares_raw']
    wave_starts = {
        '1L': {'MariTide': 7, 'W1': 47, 'W2': 85, 'W3': 123},
        '2L': {'MariTide': 25, 'W1': 65, 'W2': 103, 'W3': 141},
    }
    diseases = ['T2D', 'HFpEF', 'ASCVD', 'CKD', 'OSA', 'MASH', 'NO ORCS', 'All Others']
    ages = ['<65', '>65']
    waves = ['MariTide', 'W1', 'W2', 'W3']
    recomputed_raw = {}
    for row in raw_data:
        phases = row.get('phases', [])
        overrides = [0.0] * n_years
        if phases and any(ph['launch'] is not None for ph in phases):
            recomputed_raw[row['row']] = compute_from_diffusion(
                phases, overrides, year_dates, year_ints, n_years)
        else:
            recomputed_raw[row['row']] = np.array(row['values'][:n_years])
    normalized = []
    for line in ['1L', '2L']:
        l_label = 'L1' if line == '1L' else 'L2'
        for d_idx, disease in enumerate(diseases):
            for a_idx, age in enumerate(ages):
                offset = d_idx * 2 + a_idx
                raw_vals = {}
                for wave in waves:
                    raw_row = wave_starts[line][wave] + offset
                    raw_vals[wave] = recomputed_raw.get(raw_row, np.zeros(n_years))
                total = sum(raw_vals[w] for w in waves)
                for wave in waves:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        norm_vals = np.where(total != 0, raw_vals[wave] / total, 0.0)
                    norm_vals = np.clip(norm_vals, 0.0, 1.0)
                    d_name = 'No ORCS' if disease == 'NO ORCS' else disease
                    key = f"{d_name}{age}{wave}{l_label}"
                    normalized.append({'key': key, 'values': list(norm_vals), 'recomputed': True})
    return normalized


def lookup_input(input_data, key, n_years=29):
    key_clean = key.strip().lower()
    for row in input_data:
        if row.get('key', '').strip().lower() == key_clean:
            vals = row['values']
            arr = np.zeros(n_years)
            arr[:min(len(vals), n_years)] = vals[:n_years]
            return arr
    return np.zeros(n_years)


def lookup_excel_row(flow_data, row_num, n_years=29):
    key = f"row_{row_num}"
    if key in flow_data:
        return np.array(flow_data[key]['values'][:n_years])
    return np.zeros(n_years)


def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b != 0, a / b, 0.0)


def run_1l_2l_section(p, disease, line, flow_data, glp1_u65, glp1_o65, inputs):
    n = p['n_years']
    prefix = disease
    l_split = "L1" if line == "1L" else "L2"
    l_inj = line
    r = {}
    split_u65 = lookup_input(inputs['l1l2_split'], f"{prefix}<65", n)
    split_o65 = lookup_input(inputs['l1l2_split'], f"{prefix}>65", n)
    line_u65 = glp1_u65 * split_u65
    line_o65 = glp1_o65 * split_o65
    r['line_pts_u65'] = line_u65
    r['line_pts_o65'] = line_o65
    r['line_pts_total'] = line_u65 + line_o65
    inj_u65 = lookup_input(inputs['inj_sob'], f"{prefix}<65{l_inj}", n)
    inj_o65 = lookup_input(inputs['inj_sob'], f"{prefix}>65{l_inj}", n)
    inj_pts_u65 = line_u65 * inj_u65
    inj_pts_o65 = line_o65 * inj_o65
    r['inj_pts_u65'] = inj_pts_u65
    r['inj_pts_o65'] = inj_pts_o65
    r['inj_pts_total'] = inj_pts_u65 + inj_pts_o65
    waves = ['MariTide', 'W1', 'W2', 'W3']
    pre_access = {}
    for w in waves:
        su = lookup_input(inputs['maritide_shares'], f"{prefix}<65{w}{l_split}", n)
        so = lookup_input(inputs['maritide_shares'], f"{prefix}>65{w}{l_split}", n)
        pre_access[w] = {'pts_u65': inj_pts_u65 * su, 'pts_o65': inj_pts_o65 * so}
        pre_access[w]['pts_total'] = pre_access[w]['pts_u65'] + pre_access[w]['pts_o65']
    r['pre_access'] = pre_access
    access = {}
    for w in waves:
        ru = lookup_input(inputs['access'], f"{prefix}<65{w}{l_split}", n)
        ro = lookup_input(inputs['access'], f"{prefix}>65{w}{l_split}", n)
        access[w] = {'pts_u65': pre_access[w]['pts_u65'] * ru, 'pts_o65': pre_access[w]['pts_o65'] * ro}
        access[w]['pts_total'] = access[w]['pts_u65'] + access[w]['pts_o65']
    r['access'] = access
    sum_acc_u = sum(access[w]['pts_u65'] for w in waves)
    sum_acc_o = sum(access[w]['pts_o65'] for w in waves)
    post_access = {}
    for w in waves:
        pu = safe_div(access[w]['pts_u65'], sum_acc_u) * inj_pts_u65
        po = safe_div(access[w]['pts_o65'], sum_acc_o) * inj_pts_o65
        post_access[w] = {'pts_u65': pu, 'pts_o65': po, 'pts_total': pu + po}
    r['post_access'] = post_access
    supply = {}
    for w in waves:
        su = lookup_input(inputs['supply'], f"{prefix}<65{w}", n)
        so = lookup_input(inputs['supply'], f"{prefix}>65{w}", n)
        if np.sum(su) == 0: su = np.ones(n)
        if np.sum(so) == 0: so = np.ones(n)
        supply[w] = {'pts_u65': post_access[w]['pts_u65'] * su,
                      'pts_o65': post_access[w]['pts_o65'] * so}
        supply[w]['pts_total'] = supply[w]['pts_u65'] + supply[w]['pts_o65']
    r['supply'] = supply
    cu = lookup_input(inputs['compliance'], f"{prefix}<65{l_split}", n)
    co = lookup_input(inputs['compliance'], f"{prefix}>65{l_split}", n)
    r['mt_compliance_u65'] = supply['MariTide']['pts_u65'] * cu
    r['mt_compliance_o65'] = supply['MariTide']['pts_o65'] * co
    r['mt_compliance_total'] = r['mt_compliance_u65'] + r['mt_compliance_o65']
    pricing = p['pricing']
    upt = np.array(pricing.get('ORCs', [0]*n)[:n], dtype=float)
    if np.sum(upt) == 0: upt = np.full(n, 5.0)
    r['mt_units_total'] = r['mt_compliance_total'] * upt
    lp = np.zeros(n)
    for k, v in pricing.items():
        if 'list price' in k.lower() and 'per' not in k.lower():
            lp = np.array(v[:n]); break
    if np.sum(lp) == 0: lp = np.full(n, 100.0)
    gross_u = r['mt_compliance_u65'] * upt * lp / 1e6
    gross_o = r['mt_compliance_o65'] * upt * lp / 1e6
    r['mt_gross_total'] = gross_u + gross_o
    gtn_u = lookup_excel_row(flow_data, 268 if line == '1L' else 447, n)
    gtn_o = lookup_excel_row(flow_data, 269 if line == '1L' else 448, n)
    r['mt_net_u65'] = gross_u * (1 - gtn_u)
    r['mt_net_o65'] = gross_o * (1 - gtn_o)
    r['mt_net_total'] = r['mt_net_u65'] + r['mt_net_o65']
    sum_sup = sum(supply[w]['pts_total'] for w in waves)
    r['wave_shares'] = {w: safe_div(supply[w]['pts_total'], sum_sup) for w in waves}
    return r


def run_disease(p, disease, inputs):
    n = p['n_years']
    flow_data = p.get(f'flow_{disease}', {})
    prefix = disease
    r = {}
    bmis = ['<25', '25-27', '27-30', '30-35', '35-40', '>40']
    prev_u, prev_o = np.zeros(n), np.zeros(n)
    for b in bmis:
        prev_u += lookup_input(p['epi'], f"{prefix}<65{b}", n)
        prev_o += lookup_input(p['epi'], f"{prefix}>65{b}", n)
    r['prev_u65'], r['prev_o65'] = prev_u, prev_o
    r['prev_total'] = prev_u + prev_o
    treat_u, treat_o = np.zeros(n), np.zeros(n)
    for b in bmis:
        eu = lookup_input(p['epi'], f"{prefix}<65{b}", n)
        eo = lookup_input(p['epi'], f"{prefix}>65{b}", n)
        treat_u += eu * lookup_input(inputs['treated'], f"{prefix}<65{b}", n)
        treat_o += eo * lookup_input(inputs['treated'], f"{prefix}>65{b}", n)
    r['treated_u65'], r['treated_o65'] = treat_u, treat_o
    r['treated_total'] = treat_u + treat_o
    cov_u, cov_o = np.zeros(n), np.zeros(n)
    for b in bmis:
        eu = lookup_input(p['epi'], f"{prefix}<65{b}", n)
        eo = lookup_input(p['epi'], f"{prefix}>65{b}", n)
        tu = lookup_input(inputs['treated'], f"{prefix}<65{b}", n)
        to_ = lookup_input(inputs['treated'], f"{prefix}>65{b}", n)
        cu = lookup_input(inputs['coverage'], f"{prefix}<65{b}", n)
        co = lookup_input(inputs['coverage'], f"{prefix}>65{b}", n)
        cov_u += eu * tu * cu
        cov_o += eo * to_ * co
    r['covered_u65'], r['covered_o65'] = cov_u, cov_o
    r['covered_total'] = cov_u + cov_o
    gu = lookup_input(inputs['glp1'], f"{prefix}<65Penetration", n)
    go_ = lookup_input(inputs['glp1'], f"{prefix}>65Penetration", n)
    r['glp1_u65'] = cov_u * gu
    r['glp1_o65'] = cov_o * go_
    r['glp1_total'] = r['glp1_u65'] + r['glp1_o65']
    r['1L'] = run_1l_2l_section(p, disease, '1L', flow_data, r['glp1_u65'], r['glp1_o65'], inputs)
    r['2L'] = run_1l_2l_section(p, disease, '2L', flow_data, r['glp1_u65'], r['glp1_o65'], inputs)
    r['total_mt_net'] = r['1L']['mt_net_total'] + r['2L']['mt_net_total']
    r['total_mt_gross'] = r['1L']['mt_gross_total'] + r['2L']['mt_gross_total']
    r['total_mt_units'] = r['1L']['mt_units_total'] + r['2L']['mt_units_total']
    r['total_mt_compliance'] = r['1L']['mt_compliance_total'] + r['2L']['mt_compliance_total']
    return r


def run_full_model(p, overrides=None):
    n = p['n_years']
    year_ints = p['year_ints']
    year_dates = [datetime(yr, 1, 1) for yr in year_ints]
    params = deepcopy(p) if overrides else p
    if overrides:
        for sheet_key, key, phase_idx, param_name, value in overrides:
            for row in params[sheet_key]:
                if row['key'].strip().lower() == key.strip().lower():
                    if phase_idx < len(row.get('phases', [])):
                        row['phases'][phase_idx][param_name] = value
                    break
    inputs = {}
    for sk in ['treated', 'coverage', 'glp1', 'l1l2_split', 'inj_sob',
               'access', 'supply', 'compliance', 'cashpay']:
        inputs[sk] = recompute_input_sheet(params[sk], year_dates, year_ints, n)
    inputs['maritide_shares'] = recompute_maritide_shares(params, year_dates, year_ints, n)
    all_results = {}
    for disease in params['flow_sheets']:
        all_results[disease] = run_disease(params, disease, inputs)
    return all_results, inputs


# ─── Executive Multiplier Logic ──────────────────────────────────────────

BASE_CASES = {
    'Original': {
        'launch_year': 2029, 'peak_share_pct': 35,
        'treatment_rate_pct': 71, 'coverage_pct': 56,
        'glp1_penetration_pct': 50, 'compliance_pct': 50,
        'inj_sob_pct': 50, 'gross_price_per_unit': 113, 'gtn_pct': 50,
    },
    'Base Case 2 (Research)': {
        'launch_year': 2029, 'peak_share_pct': 25,
        'treatment_rate_pct': 36, 'coverage_pct': 50,
        'glp1_penetration_pct': 43, 'compliance_pct': 46,
        'inj_sob_pct': 60, 'gross_price_per_unit': 100, 'gtn_pct': 55,
    },
}

EXEC_DEFAULTS = BASE_CASES['Original']

RATE_SHEET_MAP = {
    # (sheet_key, base_avg_peak, start_floor_ratio)
    # start_floor_ratio: when scaling down, start is capped at peak * this ratio
    # This ensures curves ramp FROM current reality TO target peak
    'treatment_rate_pct': ('treated', 71, 0.35),   # treatment starts ~35% of peak (real-world ~12% today)
    'coverage_pct': ('coverage', 56, 0.40),          # coverage starts ~40% of peak
    'glp1_penetration_pct': ('glp1', 50, 0.30),     # GLP-1 starts ~30% of peak (still early adoption)
    'compliance_pct': ('compliance', 50, 0.50),       # compliance starts ~50% of peak
    'inj_sob_pct': ('inj_sob', 50, 0.40),            # inj SOB starts ~40% of peak
}


def apply_executive_multipliers(p, exec_mults):
    params = deepcopy(p)
    for exec_key, (sheet_key, base_avg, start_floor_ratio) in RATE_SHEET_MAP.items():
        new_pct = exec_mults.get(exec_key, base_avg)
        if new_pct == base_avg:
            continue
        mult = new_pct / base_avg
        for row in params[sheet_key]:
            phases = row.get('phases', [])
            for ph in phases:
                if ph.get('launch') is not None and ph['duration'] != 0:
                    new_peak = min(max(ph['peak'] * mult, 0.0), 1.0)
                    if mult < 1.0:
                        # When scaling DOWN, cap start so curve ramps up meaningfully
                        max_start = new_peak * start_floor_ratio
                        new_start = min(ph['start'] * mult, max_start)
                        ph['start'] = min(max(new_start, 0.0), 1.0)
                    else:
                        ph['start'] = min(max(ph['start'] * mult, 0.0), 1.0)
                    ph['peak'] = new_peak
    new_share = exec_mults.get('peak_share_pct', 35)
    if new_share != 35:
        mult = new_share / 35.0
        mt_rows = set(range(7, 23)) | set(range(25, 41))
        for row in params['maritide_shares_raw']:
            if row['row'] in mt_rows:
                for ph in row.get('phases', []):
                    if ph.get('launch') is not None and ph['duration'] != 0:
                        ph['peak'] = ph['peak'] * mult
    launch_yr = exec_mults.get('launch_year', 2029)
    if launch_yr != 2029:
        shift = launch_yr - 2029
        mt_rows = set(range(7, 23)) | set(range(25, 41))
        for row in params['maritide_shares_raw']:
            if row['row'] in mt_rows:
                phases = row.get('phases', [])
                if phases and phases[0].get('launch') is not None:
                    orig = phases[0]['launch']
                    if isinstance(orig, datetime):
                        phases[0]['launch'] = datetime(orig.year + shift, orig.month, orig.day)
    new_price = exec_mults.get('gross_price_per_unit', 113)
    if new_price != 113:
        ratio = new_price / 113.0
        for k in list(params['pricing'].keys()):
            if 'list price' in k.lower() and 'per' not in k.lower():
                params['pricing'][k] = [v * ratio for v in params['pricing'][k]]
    new_gtn = exec_mults.get('gtn_pct', 50) / 100.0
    if abs(new_gtn - 0.50) > 0.001:
        for disease in params['flow_sheets']:
            flow_key = f'flow_{disease}'
            if flow_key in params:
                for gtn_row in [268, 269, 447, 448]:
                    row_key = f"row_{gtn_row}"
                    if row_key in params[flow_key]:
                        params[flow_key][row_key]['values'] = [new_gtn] * params['n_years']
    return params


def run_scenario(p, exec_mults, detail_overrides=None):
    params = apply_executive_multipliers(p, exec_mults)
    return run_full_model(params, overrides=detail_overrides)


# ─── Key Parsing + Filtering ─────────────────────────────────────────────

DISEASES = ['T2D', 'HFpEF', 'ASCVD', 'CKD', 'OSA', 'MASH', 'No ORCS', 'All Others']
AGES = ['<65', '>65']
BMIS = ['<25', '25-27', '27-30', '30-35', '35-40', '>40']
WAVES = ['MariTide', 'W1', 'W2', 'W3']


def parse_key_parts(key):
    parts = {}
    k = key
    for d in DISEASES:
        if k.startswith(d):
            parts['disease'] = d; k = k[len(d):]; break
    for a in AGES:
        if k.startswith(a):
            parts['age'] = a; k = k[len(a):]; break
    for b in BMIS:
        if k.startswith(b):
            parts['bmi'] = b; k = k[len(b):]; break
    if 'Penetration' in k:
        k = k.replace('Penetration', '')
    for w in WAVES:
        if k.startswith(w):
            parts['wave'] = w; k = k[len(w):]; break
    for l in ['L1', 'L2', '1L', '2L']:
        if k.startswith(l) or k.endswith(l):
            parts['line'] = l; break
    return parts


def filter_keys(all_keys, filters):
    result = []
    for k in all_keys:
        parts = parse_key_parts(k)
        match = True
        for dim, vals in filters.items():
            if vals and dim in parts and parts[dim] not in vals:
                match = False; break
        if match:
            result.append(k)
    return result


# ─── Delta Formatting ────────────────────────────────────────────────────

def fmt_delta(scenario_val, base_val):
    diff = scenario_val - base_val
    if abs(base_val) > 0.01:
        pct = diff / abs(base_val) * 100
        return f"{diff:+,.0f}M ({pct:+.1f}%)"
    elif abs(diff) > 0.01:
        return f"{diff:+,.0f}M"
    return None




# ─── Sidebar Renderer ────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""<div style="font-size:0.95rem; font-weight:800; letter-spacing:0.06em;
            margin-bottom:0.5rem; text-transform:uppercase; color:#2563EB;">
            Scenario Levers
        </div>""", unsafe_allow_html=True)

        # Base case selector
        bc_names = list(BASE_CASES.keys())
        selected_bc = st.selectbox("Base Case", bc_names, key='base_case_sel',
                                    help="Switch between assumption sets. See base_case_2_assumptions.txt for details.")
        bc = BASE_CASES[selected_bc]

        # When base case changes, reset all sliders to that base case's defaults
        if 'prev_base_case' not in st.session_state:
            st.session_state['prev_base_case'] = selected_bc
        if st.session_state['prev_base_case'] != selected_bc:
            st.session_state['prev_base_case'] = selected_bc
            # Explicitly set slider values to new base case defaults
            key_map = {
                'exec_launch': 'launch_year', 'exec_peak_share': 'peak_share_pct',
                'exec_treat': 'treatment_rate_pct', 'exec_cov': 'coverage_pct',
                'exec_glp1': 'glp1_penetration_pct', 'exec_comp': 'compliance_pct',
                'exec_inj': 'inj_sob_pct', 'exec_gross_price': 'gross_price_per_unit',
                'exec_gtn': 'gtn_pct',
            }
            for sk, bk in key_map.items():
                st.session_state[sk] = bc[bk]
            st.rerun()

        with st.expander("LAUNCH & MARKET SHARE", expanded=True):
            launch_year = st.slider("MariTide Launch Year", 2027, 2035, bc['launch_year'], key='exec_launch')
            peak_share_pct = st.slider("Peak Market Share", 0, 100, bc['peak_share_pct'], 1, key='exec_peak_share',
                                        format="%d%%", help=f"Base: {bc['peak_share_pct']}%")

        with st.expander("PATIENT FUNNEL", expanded=True):
            treatment_rate_pct = st.slider("Treatment Rate", 0, 100, bc['treatment_rate_pct'], 1, key='exec_treat',
                                            format="%d%%", help=f"Base: {bc['treatment_rate_pct']}%")
            coverage_pct = st.slider("Insurance Coverage", 0, 100, bc['coverage_pct'], 1, key='exec_cov',
                                      format="%d%%", help=f"Base: {bc['coverage_pct']}%")
            glp1_pct = st.slider("GLP-1 Penetration", 0, 100, bc['glp1_penetration_pct'], 1, key='exec_glp1',
                                  format="%d%%", help=f"Base: {bc['glp1_penetration_pct']}%")
            compliance_pct = st.slider("Compliance", 0, 100, bc['compliance_pct'], 1, key='exec_comp',
                                        format="%d%%", help=f"Base: {bc['compliance_pct']}%")
            inj_sob_pct = st.slider("Injectable SOB", 0, 100, bc['inj_sob_pct'], 1, key='exec_inj',
                                     format="%d%%", help=f"Base: {bc['inj_sob_pct']}%")

        with st.expander("PRICING", expanded=True):
            gross_price = st.slider("Gross Price / Unit", 50, 300, bc['gross_price_per_unit'], 1, key='exec_gross_price',
                                     format="$%d", help=f"Base: ${bc['gross_price_per_unit']}")
            gtn_pct = st.slider("Gross-to-Net Discount", 0, 100, bc['gtn_pct'], 1, key='exec_gtn',
                                  format="%d%%", help=f"Base: {bc['gtn_pct']}%")

            ann_list = gross_price * 5
            ann_net = ann_list * (1 - gtn_pct / 100)
            st.markdown(f"""<div style="background:#F8FAFC;
                border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; margin-top:0.4rem;">
                <span style="color:#94A3B8; font-size:0.65rem; letter-spacing:0.1em; font-weight:600;">ANNUAL (5 UNITS)</span><br>
                <span style="color:#0F172A; font-size:0.9rem; font-weight:700; font-family:'JetBrains Mono',monospace;">${ann_list:,.0f}</span>
                <span style="color:#94A3B8; font-size:0.75rem;"> list</span>
                <span style="color:#CBD5E1; font-size:0.75rem;"> → </span>
                <span style="color:#2563EB; font-size:0.9rem; font-weight:700; font-family:'JetBrains Mono',monospace;">${ann_net:,.0f}</span>
                <span style="color:#64748B; font-size:0.75rem;"> net</span>
            </div>""", unsafe_allow_html=True)

        exec_mults = {
            'launch_year': launch_year, 'peak_share_pct': peak_share_pct,
            'treatment_rate_pct': treatment_rate_pct, 'coverage_pct': coverage_pct,
            'glp1_penetration_pct': glp1_pct, 'compliance_pct': compliance_pct,
            'inj_sob_pct': inj_sob_pct, 'gross_price_per_unit': gross_price, 'gtn_pct': gtn_pct,
        }

        # Active changes as pills
        st.markdown("---")
        changes = []
        labels = {
            'launch_year': 'Launch', 'peak_share_pct': 'Share', 'treatment_rate_pct': 'Treat',
            'coverage_pct': 'Coverage', 'glp1_penetration_pct': 'GLP-1', 'compliance_pct': 'Compliance',
            'inj_sob_pct': 'Inj SOB', 'gross_price_per_unit': 'Price', 'gtn_pct': 'GTN',
        }
        for k, v in exec_mults.items():
            if v != bc[k]:
                lbl = labels[k]
                if k == 'launch_year':
                    changes.append(f'<span class="change-pill">{lbl} {v}</span>')
                elif 'price' in k:
                    changes.append(f'<span class="change-pill">{lbl} ${v}</span>')
                else:
                    changes.append(f'<span class="change-pill">{lbl} {v}%</span>')

        if changes:
            st.markdown("".join(changes), unsafe_allow_html=True)
            if st.button("Reset All", key='reset_exec', type='secondary'):
                key_map = {
                    'exec_launch': 'launch_year', 'exec_peak_share': 'peak_share_pct',
                    'exec_treat': 'treatment_rate_pct', 'exec_cov': 'coverage_pct',
                    'exec_glp1': 'glp1_penetration_pct', 'exec_comp': 'compliance_pct',
                    'exec_inj': 'inj_sob_pct', 'exec_gross_price': 'gross_price_per_unit',
                    'exec_gtn': 'gtn_pct',
                }
                for sk, bk in key_map.items():
                    st.session_state[sk] = bc[bk]
                st.rerun()
        else:
            st.markdown('<div class="no-changes">Base case — no changes</div>', unsafe_allow_html=True)

    return exec_mults

# ─── Dashboard Tab ────────────────────────────────────────────────────────

def render_dashboard(p, base_results, scenario_results, has_scenario):
    years = p['years']
    n = p['n_years']
    diseases = [d for d in p['flow_sheets'] if np.sum(np.abs(base_results[d]['total_mt_net'])) > 0]

    base_net = sum(base_results[d]['total_mt_net'] for d in p['flow_sheets'])
    base_gross = sum(base_results[d]['total_mt_gross'] for d in p['flow_sheets'])
    base_pts = sum(base_results[d]['total_mt_compliance'] for d in p['flow_sheets'])

    if has_scenario:
        scen_net = sum(scenario_results[d]['total_mt_net'] for d in p['flow_sheets'])
        scen_gross = sum(scenario_results[d]['total_mt_gross'] for d in p['flow_sheets'])
        scen_pts = sum(scenario_results[d]['total_mt_compliance'] for d in p['flow_sheets'])
        show_net, show_gross, show_pts = scen_net, scen_gross, scen_pts
        show_results = scenario_results
        label = "SCENARIO"
    else:
        scen_net = scen_gross = scen_pts = None
        show_net, show_gross, show_pts = base_net, base_gross, base_pts
        show_results = base_results
        label = "BASE"

    # Status bar
    if has_scenario:
        st.markdown('<div class="status-bar"><span style="color:#2563EB; font-weight:700; letter-spacing:0.04em;">SCENARIO ACTIVE</span> <span style="color:#64748B;"> — Adjust sidebar levers. Base case shown as dotted reference.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar-base"><span style="color:#059669; font-weight:700; letter-spacing:0.04em;">BASE CASE</span> <span style="color:#64748B;"> — Matches Excel. Use sidebar to create scenarios.</span></div>', unsafe_allow_html=True)

    # ── KPI row ──
    peak_idx = np.argmax(show_net)
    idx_30, idx_35, idx_40 = 2030 - 2022, 2035 - 2022, 2040 - 2022

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak Net Revenue",
              f"${show_net[peak_idx]:,.0f}M",
              delta=fmt_delta(scen_net[peak_idx], base_net[peak_idx]) if has_scenario else f"in {years[peak_idx]}")
    k2.metric("2030 Net Revenue",
              f"${show_net[idx_30]:,.0f}M",
              delta=fmt_delta(scen_net[idx_30], base_net[idx_30]) if has_scenario else None)
    k3.metric("2035 Net Revenue",
              f"${show_net[idx_35]:,.0f}M",
              delta=fmt_delta(scen_net[idx_35], base_net[idx_35]) if has_scenario else None)
    k4.metric("2040 Patients",
              f"{show_pts[idx_40]/1e6:,.2f}M",
              delta=f"{(show_pts[idx_40]-base_pts[idx_40])/1e6:+,.2f}M" if has_scenario else None)

    st.markdown("")  # spacer

    # ── Row 1 ──
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header"><span>Revenue</span> by Disease</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=base_net, name='Base Total', showlegend=True,
            line=dict(width=2, dash='dot', color=MUTED)))
        for d in diseases:
            fig.add_trace(go.Bar(
                name=d, x=years, y=show_results[d]['total_mt_net'],
                marker_color=DISEASE_COLORS.get(d, '#888'),
                marker_line=dict(width=0)))
        fig.update_layout(**nexa_layout(yaxis_title="Net Revenue (M$)", barmode='stack'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown('<div class="section-header"><span>Revenue</span> Trend</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=base_net, name='Base Net',
                                  line=dict(width=2, dash='dot', color=MUTED)))
        fig.add_trace(go.Scatter(x=years, y=base_gross, name='Base Gross',
                                  line=dict(width=1, dash='dot', color='rgba(244,112,103,0.35)')))
        if has_scenario:
            fig.add_trace(go.Scatter(x=years, y=scen_net, name='Scenario Net',
                                      line=dict(width=3, color=GOLD),
                                      fill='tonexty', fillcolor='rgba(37,99,235,0.08)'))
            fig.add_trace(go.Scatter(x=years, y=scen_gross, name='Scenario Gross',
                                      line=dict(width=2, dash='dash', color='#F47067')))
        else:
            fig.add_trace(go.Scatter(x=years, y=base_gross, name='Gross',
                                      line=dict(width=2, color='#F47067')))
            fig.add_trace(go.Scatter(x=years, y=base_net, name='Net',
                                      line=dict(width=3, color=GOLD)))
        fig.update_layout(**nexa_layout(yaxis_title="Revenue (M$)"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Row 2 ──
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header"><span>Patient</span> Funnel</div>', unsafe_allow_html=True)
        sr = show_results
        totals = [
            ('Prevalence', sum(sr[d]['prev_total'] for d in diseases)),
            ('Treated', sum(sr[d]['treated_total'] for d in diseases)),
            ('Covered', sum(sr[d]['covered_total'] for d in diseases)),
            ('GLP-1', sum(sr[d]['glp1_total'] for d in diseases)),
            ('MariTide', sum(sr[d]['total_mt_compliance'] for d in diseases)),
        ]
        fig = go.Figure()
        for (name, vals), color in zip(totals, FUNNEL_COLORS):
            fig.add_trace(go.Scatter(x=years, y=vals / 1e6, name=name,
                                      line=dict(width=2, color=color)))
        base_comp = sum(base_results[d]['total_mt_compliance'] for d in diseases)
        fig.add_trace(go.Scatter(x=years, y=base_comp / 1e6, name='Base MariTide',
                                  line=dict(width=2, dash='dot', color=MUTED)))
        fig.update_layout(**nexa_layout(yaxis_title="Patients (M)"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c4:
        st.markdown('<div class="section-header"><span>Wave</span> Shares (1L)</div>', unsafe_allow_html=True)
        waves = ['MariTide', 'W1', 'W2', 'W3']
        wave_pts = {w: sum(sr[d]['1L']['supply'].get(w, {}).get('pts_total', np.zeros(n))
                         for d in diseases) for w in waves}
        total_supply = sum(wave_pts[w] for w in waves)
        fig = go.Figure()
        for w in waves:
            share = safe_div(wave_pts[w], total_supply) * 100
            fig.add_trace(go.Scatter(x=years, y=share, name=w, stackgroup='one',
                                      line=dict(width=0), fillcolor=WAVE_COLORS[w]))
        base_wave = {w: sum(base_results[d]['1L']['supply'].get(w, {}).get('pts_total', np.zeros(n))
                          for d in diseases) for w in waves}
        base_total = sum(base_wave[w] for w in waves)
        base_mt_share = safe_div(base_wave['MariTide'], base_total) * 100
        fig.add_trace(go.Scatter(x=years, y=base_mt_share, name='Base MT',
                                  line=dict(width=2, dash='dot', color='white')))
        fig.update_layout(**nexa_layout(yaxis_title="Share (%)", yaxis_range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Row 3: Table + Delta/Pie ──
    c5, c6 = st.columns(2)

    with c5:
        st.markdown('<div class="section-header"><span>Revenue</span> Summary (M$)</div>', unsafe_allow_html=True)
        table_years = [2029, 2030, 2032, 2035, 2040, 2045, 2050]
        rows = []
        for d in diseases:
            row = {'Disease': d}
            for yr in table_years:
                idx = yr - 2022
                bv = base_results[d]['total_mt_net'][idx]
                if has_scenario:
                    sv = scenario_results[d]['total_mt_net'][idx]
                    row[str(yr)] = f"{sv:,.0f} ({sv-bv:+,.0f})"
                else:
                    row[str(yr)] = f"{bv:,.0f}"
            rows.append(row)
        tot = {'Disease': 'TOTAL'}
        for yr in table_years:
            idx = yr - 2022
            bv = base_net[idx]
            if has_scenario:
                sv = scen_net[idx]
                tot[str(yr)] = f"{sv:,.0f} ({sv-bv:+,.0f})"
            else:
                tot[str(yr)] = f"{bv:,.0f}"
        rows.append(tot)
        st.dataframe(pd.DataFrame(rows).set_index('Disease'), use_container_width=True)

    with c6:
        if has_scenario:
            st.markdown('<div class="section-header"><span>Delta</span> vs Base</div>', unsafe_allow_html=True)
            delta_net = scen_net - base_net
            fig = go.Figure()
            colors = [GOLD if v >= 0 else '#F47067' for v in delta_net]
            fig.add_trace(go.Bar(x=years, y=delta_net, marker_color=colors,
                                  marker_line=dict(width=0)))
            fig.add_hline(y=0, line_dash='dash', line_color='#E2E8F0', line_width=1)
            fig.update_layout(**nexa_layout(yaxis_title="Delta (M$)"))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown('<div class="section-header">2035 <span>Revenue Mix</span></div>', unsafe_allow_html=True)
            idx_35 = 2035 - 2022
            vals_35 = {d: max(base_results[d]['total_mt_net'][idx_35], 0) for d in diseases}
            vals_35 = {k: v for k, v in vals_35.items() if v > 0}
            if vals_35:
                fig = go.Figure(data=[go.Pie(
                    labels=list(vals_35.keys()), values=list(vals_35.values()),
                    hole=0.45, textinfo='label+percent',
                    textfont=dict(color=LIGHT, size=11),
                    marker=dict(colors=[DISEASE_COLORS.get(d, '#888') for d in vals_35.keys()],
                                line=dict(color='#FFFFFF', width=2)))])
                fig.update_layout(height=400, margin=dict(t=10, b=10),
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color=LIGHT),
                                  legend=dict(font=dict(color=MUTED, size=10)))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Waterfall ──
    st.markdown('<div class="section-header"><span>Patient</span> Pipeline Waterfall</div>', unsafe_allow_html=True)
    wc1, wc2 = st.columns([1, 1])
    sel_disease = wc1.selectbox("Disease", diseases, key='wf_dis', label_visibility='collapsed')
    sel_year = wc2.selectbox("Year", [2029, 2030, 2032, 2035, 2040, 2045, 2050],
                              index=3, key='wf_yr', label_visibility='collapsed')
    idx = sel_year - 2022
    res = show_results[sel_disease]
    bres = base_results[sel_disease]

    stages = ['Prevalence', 'Treated', 'Covered', 'GLP-1', '1L Line', '1L Inj SOB',
              '1L MT Pre-Acc', '1L MT Post-Acc', '1L MT Supply', '1L MariTide']
    vals = [res['prev_total'][idx], res['treated_total'][idx], res['covered_total'][idx],
            res['glp1_total'][idx], res['1L']['line_pts_total'][idx],
            res['1L']['inj_pts_total'][idx],
            res['1L']['pre_access']['MariTide']['pts_total'][idx],
            res['1L']['post_access']['MariTide']['pts_total'][idx],
            res['1L']['supply']['MariTide']['pts_total'][idx],
            res['1L']['mt_compliance_total'][idx]]
    bvals = [bres['prev_total'][idx], bres['treated_total'][idx], bres['covered_total'][idx],
             bres['glp1_total'][idx], bres['1L']['line_pts_total'][idx],
             bres['1L']['inj_pts_total'][idx],
             bres['1L']['pre_access']['MariTide']['pts_total'][idx],
             bres['1L']['post_access']['MariTide']['pts_total'][idx],
             bres['1L']['supply']['MariTide']['pts_total'][idx],
             bres['1L']['mt_compliance_total'][idx]]

    bar_colors = [FUNNEL_COLORS[0]]*4 + [FUNNEL_COLORS[3]]*2 + [FUNNEL_COLORS[1]]*2 + [FUNNEL_COLORS[2], GOLD]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stages, y=[v / 1e6 for v in vals], marker_color=bar_colors,
        marker_line=dict(width=0),
        text=[f"{v/1e6:,.2f}M" for v in vals], textposition='outside',
        textfont=dict(color=LIGHT, size=9), name=sel_disease))
    fig.add_trace(go.Bar(
        x=stages, y=[v / 1e6 for v in bvals],
        marker_color='rgba(125,133,144,0.2)', marker_line=dict(width=0),
        text=[f"{v/1e6:,.2f}M" for v in bvals], textposition='outside',
        textfont=dict(color='#94A3B8', size=8), name='Base'))
    wl = nexa_layout(yaxis_title="Patients (M)", height=400, barmode='group')
    wl['xaxis'] = dict(tickangle=-35, gridcolor='#E2E8F0', tickfont=dict(color=MUTED, size=9))
    fig.update_layout(**wl)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Cumulative ──
    st.markdown('<div class="section-header"><span>Cumulative</span> Net Revenue</div>', unsafe_allow_html=True)
    cum_base = np.cumsum(base_net)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=cum_base, name='Base',
                              line=dict(width=2, dash='dot', color=MUTED)))
    if has_scenario:
        cum_scen = np.cumsum(scen_net)
        fig.add_trace(go.Scatter(x=years, y=cum_scen, name='Scenario',
                                  line=dict(width=3, color=GOLD),
                                  fill='tonexty', fillcolor='rgba(37,99,235,0.06)'))
    else:
        fig.add_trace(go.Scatter(x=years, y=cum_base, name='Cumulative',
                                  line=dict(width=3, color=GOLD)))
    fig.update_layout(**nexa_layout(height=320, yaxis_title="Cumulative (M$)"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})






# ─── Detailed Controls Tab ────────────────────────────────────────────────

SHAPE_OPTIONS = ["S-Curve", "Moderate", "Blended", "Steep", "Exponential"]
SHAPE_TO_VAL = {"S-Curve": 1.0, "Moderate": 1.25, "Blended": 1.5, "Steep": 1.75, "Exponential": 2.0}
SHAPE_STEPS = [1.0, 1.25, 1.5, 1.75, 2.0]


def val_to_shape_name(v):
    rounded = round(v * 4) / 4
    rounded = min(max(rounded, 1.0), 2.0)
    mapping = {1.0: "S-Curve", 1.25: "Moderate", 1.5: "Blended", 1.75: "Steep", 2.0: "Exponential"}
    return mapping.get(rounded, "Blended")


def snap_shape(v):
    return min(SHAPE_STEPS, key=lambda x: abs(x - v))


def get_last_active_phase(row):
    if not row.get('phases'):
        return None, None
    for i in reversed(range(len(row['phases']))):
        ph = row['phases'][i]
        if ph.get('launch') is not None and (ph['duration'] != 0 or ph['peak'] != 0):
            return i, ph
    return None, None


DETAIL_SECTIONS = [
    {
        'group': 'Patient Funnel',
        'sections': [
            {'name': 'Treatment Rate', 'key': 'treated', 'desc': '% of prevalent patients pharmacologically treated',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
            {'name': 'Coverage', 'key': 'coverage', 'desc': 'Insurance coverage rate for treated patients',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
            {'name': 'GLP-1 Penetration', 'key': 'glp1', 'desc': 'GLP-1 class penetration among covered patients',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
        ]
    },
    {
        'group': 'Line & Formulation',
        'sections': [
            {'name': '1L / 2L Split', 'key': 'l1l2_split', 'desc': 'First-line vs second-line therapy allocation',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
            {'name': 'Injectable SOB', 'key': 'inj_sob', 'desc': 'Injectable share of business',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
        ]
    },
    {
        'group': 'Share & Access',
        'sections': [
            {'name': 'MariTide Shares', 'key': 'maritide_shares_raw', 'desc': 'Raw wave share diffusion curves (pre-normalization)',
             'editable': False, 'is_rate': False, 'group_by': None},
            {'name': 'Access', 'key': 'access', 'desc': 'Payer access and formulary listing rates',
             'editable': True, 'is_rate': True, 'group_by': 'wave'},
            {'name': 'Supply', 'key': 'supply', 'desc': 'Supply availability and fulfillment rates',
             'editable': True, 'is_rate': True, 'group_by': 'wave'},
        ]
    },
    {
        'group': 'Persistence & Pricing',
        'sections': [
            {'name': 'Compliance', 'key': 'compliance', 'desc': 'Patient compliance and persistency on therapy',
             'editable': True, 'is_rate': True, 'group_by': 'disease'},
            {'name': 'Pricing', 'key': 'pricing', 'desc': 'Price per unit, GTN, and related inputs (adjust via sidebar)',
             'editable': False, 'is_rate': False, 'group_by': None},
        ]
    },
    {
        'group': 'Reference',
        'sections': [
            {'name': 'Epidemiology', 'key': 'epi', 'desc': 'Prevalence by disease, age, and BMI (read-only)',
             'editable': False, 'is_rate': False, 'group_by': None},
        ]
    },
]

GROUP_LABELS = {'disease': 'Disease', 'wave': 'Wave'}
GROUP_COLORS = {'disease': DISEASE_COLORS, 'wave': WAVE_COLORS}
GROUP_VALUES = {'disease': DISEASES, 'wave': WAVES}


def build_group_summary(p, sheet_key, group_by):
    """Build disease-level or wave-level summary of a rate sheet."""
    sheet_data = p[sheet_key]
    group_values = GROUP_VALUES[group_by]

    active_rows = [r for r in sheet_data
                   if r.get('phases') and any(ph.get('launch') is not None for ph in r['phases'])]
    if not active_rows:
        return []

    groups = {}
    for row in active_rows:
        parts = parse_key_parts(row['key'])
        gval = parts.get(group_by, 'Other')
        if gval not in groups:
            groups[gval] = []
        groups[gval].append(row)

    summary = []
    for gval in group_values:
        if gval not in groups:
            continue
        rows = groups[gval]
        peaks, shapes, durs = [], [], []
        for row in rows:
            _, ph = get_last_active_phase(row)
            if ph:
                peaks.append(ph['peak'])
                shapes.append(ph['shape'])
                durs.append(ph['duration'])
        if not peaks:
            continue
        summary.append({
            'label': gval,
            'avg_peak': np.mean(peaks),
            'avg_shape': np.mean(shapes),
            'avg_dur': np.mean(durs),
            'count': len(peaks),
            'rows': rows,
        })
    return summary


def compute_group_curve(rows, p, peak_override=None, shape_override=None, dur_override=None):
    """Compute averaged diffusion curve for a group of rows, with optional overrides."""
    year_ints = p['year_ints']
    year_dates = [datetime(yr, 1, 1) for yr in year_ints]
    nn = p['n_years']

    # Compute peak scale factor from last-phase averages
    if peak_override is not None:
        orig_peaks = []
        for row in rows:
            _, ph = get_last_active_phase(row)
            if ph:
                orig_peaks.append(ph['peak'])
        orig_avg_peak = np.mean(orig_peaks) if orig_peaks else 0
    else:
        orig_avg_peak = 0

    curves = []
    for row in rows:
        phases = row['phases']
        if peak_override is not None or shape_override is not None or dur_override is not None:
            phases = deepcopy(phases)
            # Apply overrides to ALL active phases, not just the last one
            for ph in phases:
                if ph.get('launch') is not None and (ph['duration'] != 0 or ph['peak'] != 0):
                    if peak_override is not None and orig_avg_peak > 0:
                        scale = peak_override / orig_avg_peak
                        ph['peak'] = min(max(ph['peak'] * scale, 0.0), 1.0)
                        ph['start'] = min(max(ph['start'] * scale, 0.0), 1.0)
                    if shape_override is not None:
                        ph['shape'] = shape_override
                    if dur_override is not None:
                        ph['duration'] = dur_override
        vals = compute_from_diffusion(
            phases, row.get('overrides', [0.0] * nn), year_dates, year_ints, nn)
        curves.append(vals)
    return np.mean(curves, axis=0) if curves else np.zeros(nn)


def collect_group_overrides(sheet_key, rows, orig_avg_peak, orig_avg_shape, orig_avg_dur,
                            new_peak, new_shape, new_dur):
    """Generate override tuples for all rows in a group based on slider changes."""
    overrides = []
    peak_changed = abs(new_peak - orig_avg_peak) > 0.001
    shape_changed = abs(new_shape - orig_avg_shape) > 0.05
    dur_changed = abs(new_dur - orig_avg_dur) > 0.05

    if not (peak_changed or shape_changed or dur_changed):
        return overrides

    for row in rows:
        if not row.get('phases'):
            continue
        # Apply to ALL active phases, not just the last one
        for ph_idx, ph in enumerate(row['phases']):
            if ph.get('launch') is None or (ph['duration'] == 0 and ph['peak'] == 0):
                continue
            if peak_changed and orig_avg_peak > 0:
                scale = new_peak / orig_avg_peak
                overrides.append((sheet_key, row['key'], ph_idx, 'peak',
                                  min(max(ph['peak'] * scale, 0.0), 1.0)))
                overrides.append((sheet_key, row['key'], ph_idx, 'start',
                                  min(max(ph['start'] * scale, 0.0), 1.0)))
            if shape_changed:
                overrides.append((sheet_key, row['key'], ph_idx, 'shape', new_shape))
            if dur_changed:
                overrides.append((sheet_key, row['key'], ph_idx, 'duration', new_dur))
    return overrides


def render_rate_section(p, sec_cfg, sec_id):
    """Render rate section: clickable summary table + interactive curve panel."""
    sheet_key = sec_cfg['key']
    group_by = sec_cfg.get('group_by', 'disease')
    group_label = GROUP_LABELS[group_by]
    color_map = GROUP_COLORS[group_by]

    summary = build_group_summary(p, sheet_key, group_by)
    if not summary:
        st.caption("No parameterized data.")
        return []

    col_table, col_curve = st.columns([2, 3])

    with col_table:
        # Build display dataframe
        df_display = pd.DataFrame([{
            group_label: s['label'],
            'Peak (%)': round(s['avg_peak'] * 100, 1),
            'Curve': val_to_shape_name(s['avg_shape']),
            'Duration': f"{s['avg_dur']:.1f} yr",
            'Params': s['count'],
        } for s in summary])

        event = st.dataframe(
            df_display,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            key=f'select_{sec_id}',
        )

        selected_rows = event.selection.rows if event.selection else []

        if not selected_rows:
            st.markdown("""<div style="color:#94A3B8; font-size:0.78rem; text-align:center;
                padding:1rem; border:1px dashed #CBD5E1; border-radius:8px; margin-top:0.5rem;
                background:#FAFBFC;">
                Click a row to open the curve editor
            </div>""", unsafe_allow_html=True)

    with col_curve:
        if not selected_rows:
            # Show all-groups overlay as default
            fig = go.Figure()
            for srow in summary:
                base_curve = compute_group_curve(srow['rows'], p) * 100
                fig.add_trace(go.Scatter(
                    x=p['years'], y=base_curve, name=srow['label'],
                    line=dict(width=2, color=color_map.get(srow['label'], '#888'))))
            fig.update_layout(**nexa_layout(
                height=350, yaxis_title=f"{sec_cfg['name']} (%)",
                yaxis_range=[0, None]))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            sel_idx = selected_rows[0]
            srow = summary[sel_idx]
            gval = srow['label']
            color = color_map.get(gval, GOLD)

            # Header
            st.markdown(f"""<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                <span style="color:{color}; font-size:1rem; font-weight:800;">{gval}</span>
                <span style="color:#94A3B8; font-size:0.75rem;">({srow['count']} parameters)</span>
            </div>""", unsafe_allow_html=True)

            # Sliders for curve control
            default_peak = round(srow['avg_peak'] * 100, 1)
            default_shape = snap_shape(srow['avg_shape'])
            default_dur = round(srow['avg_dur'], 1)

            new_peak_pct = st.slider(
                "Peak Rate", 0.0, 100.0, default_peak, 0.5,
                format="%.1f%%", key=f'{sec_id}_{gval}_peak',
                help="Steady-state rate — scales all underlying parameters proportionally")

            new_shape = st.select_slider(
                "Curve Shape",
                options=SHAPE_STEPS,
                value=default_shape,
                format_func=lambda x: f"{'S-Curve' if x == 1.0 else 'Moderate' if x == 1.25 else 'Blended' if x == 1.5 else 'Steep' if x == 1.75 else 'Exponential'}",
                key=f'{sec_id}_{gval}_shape',
                help="S-Curve = gradual ramp → Exponential = fast saturation")

            new_dur = st.slider(
                "Time to Peak", 1.0, 30.0, default_dur, 0.5,
                format="%.1f yr", key=f'{sec_id}_{gval}_dur',
                help="Years to reach the peak rate")

            # Compute curves
            base_curve = compute_group_curve(srow['rows'], p) * 100

            has_changes = (abs(new_peak_pct / 100 - srow['avg_peak']) > 0.001 or
                           abs(new_shape - srow['avg_shape']) > 0.05 or
                           abs(new_dur - srow['avg_dur']) > 0.05)

            if has_changes:
                scen_curve = compute_group_curve(
                    srow['rows'], p,
                    peak_override=new_peak_pct / 100,
                    shape_override=new_shape if abs(new_shape - srow['avg_shape']) > 0.05 else None,
                    dur_override=new_dur if abs(new_dur - srow['avg_dur']) > 0.05 else None,
                ) * 100
            else:
                scen_curve = base_curve

            # Chart — zoom to active range for visibility
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=p['years'], y=base_curve, name='Base',
                line=dict(width=2, dash='dot', color=MUTED),
                fill='tozeroy', fillcolor='rgba(125,133,144,0.03)'))
            fig.add_trace(go.Scatter(
                x=p['years'], y=scen_curve, name='Scenario',
                line=dict(width=3, color=color),
                fill='tonexty', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)' if color.startswith('#') and len(color) == 7 else 'rgba(37,99,235,0.08)'))

            # Auto-zoom: find first year where curve > 1% and crop x-axis
            active_mask = (base_curve > 0.5) | (scen_curve > 0.5)
            if np.any(active_mask):
                first_active = p['year_ints'][np.argmax(active_mask)]
                x_start = max(first_active - 2, 2022)
            else:
                x_start = 2027
            # Auto y-range: zoom to the data range for better visibility
            y_vals = np.concatenate([base_curve[base_curve > 0], scen_curve[scen_curve > 0]]) if np.any(base_curve > 0) else np.array([0, 100])
            y_min_val = max(np.min(y_vals) - 5, 0) if len(y_vals) > 0 else 0
            y_max_val = min(np.max(y_vals) * 1.1 + 2, 105) if len(y_vals) > 0 else 105

            fig.update_layout(**nexa_layout(
                height=280,
                yaxis_title=f"{sec_cfg['name']} (%)",
                xrange=[x_start, 2050.5],
                yaxis_range=[y_min_val, y_max_val]))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Change indicator
            if has_changes:
                delta_peak = new_peak_pct - default_peak
                st.markdown(f"""<div style="background:#EFF6FF;
                    border:1px solid #BFDBFE; border-radius:8px; padding:0.5rem 0.8rem; font-size:0.78rem;">
                    <span style="color:#2563EB; font-weight:600;">Override active</span>
                    <span style="color:#64748B;"> — Peak {delta_peak:+.1f}pp</span>
                </div>""", unsafe_allow_html=True)

    # Collect overrides from ALL groups (not just selected) using session state
    all_overrides = []
    for srow in summary:
        gval = srow['label']
        peak_key = f'{sec_id}_{gval}_peak'
        shape_key = f'{sec_id}_{gval}_shape'
        dur_key = f'{sec_id}_{gval}_dur'

        stored_peak = st.session_state.get(peak_key)
        stored_shape = st.session_state.get(shape_key)
        stored_dur = st.session_state.get(dur_key)

        if stored_peak is None and stored_shape is None and stored_dur is None:
            continue

        new_p = (stored_peak / 100) if stored_peak is not None else srow['avg_peak']
        new_s = stored_shape if stored_shape is not None else srow['avg_shape']
        new_d = stored_dur if stored_dur is not None else srow['avg_dur']

        ovr = collect_group_overrides(
            sheet_key, srow['rows'],
            srow['avg_peak'], srow['avg_shape'], srow['avg_dur'],
            new_p, new_s, new_d)
        all_overrides.extend(ovr)

    return all_overrides


def render_shares_section(p, sec_id):
    """Render MariTide shares as a read-only wave summary with curves."""
    sheet_data = p['maritide_shares_raw']
    year_ints = p['year_ints']
    year_dates = [datetime(yr, 1, 1) for yr in year_ints]
    nn = p['n_years']

    wave_labels = {range(7, 23): 'MariTide 1L', range(25, 41): 'MariTide 2L',
                   range(47, 63): 'W1 1L', range(65, 81): 'W1 2L',
                   range(85, 101): 'W2 1L', range(103, 119): 'W2 2L',
                   range(123, 139): 'W3 1L', range(141, 157): 'W3 2L'}

    wave_summary = {}
    for row in sheet_data:
        if not row.get('phases'):
            continue
        label = None
        for rng, lbl in wave_labels.items():
            if row['row'] in rng:
                label = lbl
                break
        if not label:
            continue
        if label not in wave_summary:
            wave_summary[label] = {'peaks': [], 'rows': []}
        _, ph = get_last_active_phase(row)
        if ph:
            wave_summary[label]['peaks'].append(ph['peak'])
            wave_summary[label]['rows'].append(row)

    col_t, col_c = st.columns([2, 3])

    with col_t:
        rows = []
        for label in ['MariTide 1L', 'MariTide 2L', 'W1 1L', 'W1 2L', 'W2 1L', 'W2 2L', 'W3 1L', 'W3 2L']:
            if label in wave_summary:
                info = wave_summary[label]
                rows.append({
                    'Wave': label,
                    'Avg Peak (%)': f"{np.mean(info['peaks'])*100:.1f}%",
                    'Params': len(info['peaks']),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_c:
        fig = go.Figure()
        wave_colors = {'MariTide': GOLD, 'W1': '#F47067', 'W2': '#58A6FF', 'W3': '#BC8CFF'}
        line_dashes = {'1L': 'solid', '2L': 'dash'}
        for label, info in wave_summary.items():
            wave_name = label.split(' ')[0]
            line_type = label.split(' ')[1]
            curves = []
            for row in info['rows']:
                vals = compute_from_diffusion(
                    row['phases'], [0.0] * nn, year_dates, year_ints, nn)
                curves.append(vals)
            avg_curve = np.mean(curves, axis=0) * 100
            fig.add_trace(go.Scatter(
                x=p['years'], y=avg_curve, name=label,
                line=dict(width=2.5 if wave_name == 'MariTide' else 1.5,
                          color=wave_colors.get(wave_name, '#888'),
                          dash=line_dashes.get(line_type, 'solid'))))
        fig.update_layout(**nexa_layout(height=350, yaxis_title="Raw Share (%)"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_pricing_section(p, sec_id):
    """Render pricing as a clean read-only table."""
    st.markdown("""<div style="background:#EFF6FF;
        border:1px solid #BFDBFE; border-radius:8px; padding:0.6rem 0.9rem;
        margin-bottom:0.8rem; font-size:0.8rem;">
        <span style="color:#2563EB; font-weight:600;">Tip:</span>
        <span style="color:#64748B;"> Adjust pricing via the sidebar Gross Price and GTN sliders</span>
    </div>""", unsafe_allow_html=True)

    rows = []
    for k, v in p['pricing'].items():
        vals = v if isinstance(v, list) else [v]
        row = {'Parameter': k}
        for yr in [2029, 2030, 2035, 2040, 2050]:
            idx = yr - 2022
            if idx < len(vals):
                row[str(yr)] = f"${vals[idx]:,.1f}" if 'price' in k.lower() else f"{vals[idx]:,.2f}"
            else:
                row[str(yr)] = "—"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index('Parameter'), use_container_width=True)


def render_epi_section(p, sec_id):
    """Render epidemiology as a read-only prevalence summary by disease."""
    sheet_data = p['epi']
    year_ints = p['year_ints']
    year_dates = [datetime(yr, 1, 1) for yr in year_ints]
    nn = p['n_years']
    display_years = [2025, 2030, 2035, 2040, 2050]

    disease_totals = {}
    for disease in DISEASES:
        total = np.zeros(nn)
        count = 0
        for row in sheet_data:
            if row.get('key', '').startswith(disease):
                phases = row.get('phases', [])
                overrides_list = row.get('overrides', [0.0] * nn)
                if phases and any(ph.get('launch') is not None for ph in phases):
                    vals = compute_from_diffusion(phases, overrides_list, year_dates, year_ints, nn)
                else:
                    vals = np.array(row['values'][:nn]) if len(row.get('values', [])) >= nn else np.zeros(nn)
                total += vals
                count += 1
        if count > 0:
            disease_totals[disease] = total

    col_t, col_c = st.columns([2, 3])
    with col_t:
        rows = []
        for disease in DISEASES:
            if disease not in disease_totals:
                continue
            vals = disease_totals[disease]
            row = {'Disease': disease}
            for yr in display_years:
                idx = yr - 2022
                v = vals[idx]
                if abs(v) >= 1e6:
                    row[str(yr)] = f"{v/1e6:,.1f}M"
                elif abs(v) >= 1e3:
                    row[str(yr)] = f"{v/1e3:,.0f}K"
                else:
                    row[str(yr)] = f"{v:,.0f}"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index('Disease'), use_container_width=True)

    with col_c:
        fig = go.Figure()
        for disease in DISEASES:
            if disease in disease_totals:
                fig.add_trace(go.Scatter(
                    x=p['years'], y=disease_totals[disease] / 1e6, name=disease,
                    line=dict(width=2, color=DISEASE_COLORS.get(disease, '#888'))))
        fig.update_layout(**nexa_layout(height=350, yaxis_title="Prevalence (M)"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_detailed_controls(p):
    """Render all detailed controls as stacked collapsible sections."""
    all_overrides = []
    sec_counter = 0

    for group in DETAIL_SECTIONS:
        st.markdown(f"""<div style="font-size:0.78rem; font-weight:700;
            letter-spacing:0.1em; text-transform:uppercase; margin:1.4rem 0 0.5rem 0;
            padding-bottom:0.4rem; border-bottom:1px solid #E2E8F0; color:#2563EB;">
            {group['group']}
        </div>""", unsafe_allow_html=True)

        for sec_cfg in group['sections']:
            sec_id = f"sec{sec_counter}"
            sec_counter += 1
            sheet_key = sec_cfg['key']

            if sheet_key == 'pricing':
                badge = f"{len(p.get('pricing', {}))} params"
            elif sheet_key == 'maritide_shares_raw':
                badge = "8 wave curves"
            elif sheet_key == 'epi':
                badge = f"{len(DISEASES)} diseases"
            elif sec_cfg.get('group_by') == 'wave':
                badge = f"{len(WAVES)} waves"
            else:
                badge = f"{len(DISEASES)} diseases"

            with st.expander(f"{sec_cfg['name']}  —  {badge}", expanded=False):
                st.markdown(f'<div style="color:#64748B; font-size:0.78rem; margin-bottom:0.6rem;">{sec_cfg["desc"]}</div>',
                            unsafe_allow_html=True)

                if sheet_key == 'pricing':
                    render_pricing_section(p, sec_id)
                elif sheet_key == 'maritide_shares_raw':
                    render_shares_section(p, sec_id)
                elif sheet_key == 'epi':
                    render_epi_section(p, sec_id)
                elif sec_cfg['editable'] and sec_cfg['is_rate']:
                    ovr = render_rate_section(p, sec_cfg, sec_id)
                    if ovr:
                        all_overrides.extend(ovr)

    return all_overrides if all_overrides else None


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown("""<div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:-0.3rem;">
        <span style="color:#0F172A; font-size:1.7rem; font-weight:900; letter-spacing:-0.03em;">MariTide</span>
        <span style="color:#2563EB; font-size:1.1rem; font-weight:700; letter-spacing:0.02em;">PharmaACE</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style="color:#94A3B8; font-size:0.72rem; margin-bottom:0.5rem;
        font-family: 'JetBrains Mono', monospace; letter-spacing:0.05em;">
        ANNUAL 2022-2050  //  8 DISEASES  //  DIFFUSION-RECOMPUTED
    </div>""", unsafe_allow_html=True)

    # ── Excel Upload ──
    with st.expander("📂  Update model — upload a new Excel workbook (.xlsm / .xlsx)", expanded=False):
        st.markdown("""<div style="color:#64748B; font-size:0.82rem; margin-bottom:0.6rem;">
            Upload a new version of the MariTide PACE Excel workbook to refresh all model parameters.
            The dashboard will update automatically once the file is processed.
        </div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop Excel file here", type=["xlsm", "xlsx"],
            label_visibility="collapsed", key="excel_upload")
        if uploaded_file is not None:
            st.session_state["uploaded_excel_bytes"] = uploaded_file.read()
            st.success(f"✓ Loaded: {uploaded_file.name}  —  parameters will refresh below.")

    # Load params: from uploaded Excel if available, else from bundled pkl
    if "uploaded_excel_bytes" in st.session_state:
        p = extract_params_from_excel(st.session_state["uploaded_excel_bytes"])
        st.markdown("""<div style="background:#F0FDF4; border-left:3px solid #059669;
            padding:0.4rem 1rem; border-radius:0 6px 6px 0; font-size:0.78rem; margin-bottom:0.6rem;">
            <span style="color:#059669; font-weight:700;">LIVE EXCEL</span>
            <span style="color:#64748B;"> — dashboard is running from your uploaded workbook</span>
        </div>""", unsafe_allow_html=True)
    else:
        p = load_params()

    base_results, base_inputs = compute_base_case(p)

    exec_mults = render_sidebar()
    exec_changed = any(exec_mults[k] != EXEC_DEFAULTS[k] for k in EXEC_DEFAULTS)

    tab1, tab2 = st.tabs(["Dashboard", "Detailed Controls"])

    with tab2:
        detail_overrides = render_detailed_controls(p)

    has_detail = detail_overrides is not None and len(detail_overrides) > 0
    has_scenario = exec_changed or has_detail

    if has_scenario:
        with st.spinner("Computing scenario..."):
            scenario_results, _ = run_scenario(
                p, exec_mults,
                detail_overrides=detail_overrides if has_detail else None)
    else:
        scenario_results = None

    with tab1:
        render_dashboard(p, base_results, scenario_results, has_scenario)


if __name__ == '__main__':
    main()
