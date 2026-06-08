import math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

st.set_page_config(
    page_title="StockMind · Neural Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e6e0; }
[data-testid="stSidebar"] { background-color: #0f0f18; border-right: 1px solid #1e1e2e; }
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; color: #c8ff00 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #666 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] { background: #12121c; border: 1px solid #1e1e2e; border-radius: 8px; padding: 1rem; }
[data-testid="stTabs"] button { font-family: 'DM Mono', monospace; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; color: #555; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #c8ff00; border-bottom: 2px solid #c8ff00; background: transparent; }
.stButton > button { background: #c8ff00; color: #0a0a0f; border: none; border-radius: 4px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 13px; letter-spacing: 0.05em; padding: 0.6rem 2rem; width: 100%; }
.stButton > button:hover { background: #d4ff33; transform: translateY(-1px); }
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input { background: #12121c !important; border: 1px solid #1e1e2e !important; color: #e8e6e0 !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; }
hr { border-color: #1e1e2e !important; }
.stAlert { background: #12121c !important; border: 1px solid #1e1e2e !important; border-radius: 8px !important; color: #888 !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
.stProgress > div > div { background: #c8ff00 !important; }
.streamlit-expanderHeader { background: #12121c !important; border: 1px solid #1e1e2e !important; color: #888 !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; color: #e8e6e0; margin: 0; }
.hero-accent { color: #c8ff00; }
.mono { font-family: 'DM Mono', monospace; font-size: 12px; color: #555; }
</style>
""", unsafe_allow_html=True)

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0f", plot_bgcolor="#0f0f18",
    font=dict(family="DM Mono, monospace", color="#888", size=11),
    xaxis=dict(gridcolor="#1a1a2a", showgrid=True, zeroline=False,
               tickfont=dict(size=10), linecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1a1a2a", showgrid=True, zeroline=False,
               tickfont=dict(size=10), linecolor="#1e1e2e"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e",
                borderwidth=1, font=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified"
)

# ── Data ──
@st.cache_data(ttl=300)
def download_stock(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.reset_index()

def add_features(df, use_advanced=True):
    df = df.copy()
    df['Close'] = df['Close'].astype(float)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [5, 10, 20]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
        df[f'STD_{w}'] = df['Close'].rolling(w).std()
    df['VolChange'] = df['Volume'].pct_change() if 'Volume' in df.columns else 0.0
    if use_advanced:
        delta = df['Close'].diff()
        up = delta.clip(lower=0); dn = -delta.clip(upper=0)
        eu = up.ewm(com=13, adjust=False).mean(); ed = dn.ewm(com=13, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + eu / (ed + 1e-9)))
        e12 = df['Close'].ewm(span=12, adjust=False).mean()
        e26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = e12 - e26
        df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
        ma20 = df['Close'].rolling(20).mean(); s20 = df['Close'].rolling(20).std()
        df['BB_upper'] = ma20 + 2*s20; df['BB_lower'] = ma20 - 2*s20
    return df.dropna().reset_index(drop=True)

def make_supervised(df, feat_cols, horizon):
    X, y = [], []
    for i in range(len(df) - horizon):
        X.append(df.loc[i, feat_cols].values.astype(float))
        y.append(float(df.loc[i + horizon, 'Close']))
    return np.vstack(X), np.array(y).reshape(-1, 1)

# ── Models ──
class BaselineMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): return self.net(x)

class OptimizedMLP(nn.Module):
    def __init__(self, d, drop=0.05):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.net(x)

class EnhancedMLP(nn.Module):
    def __init__(self, d, drop=0.05):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(128,64), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(64,1))
    def forward(self, x): return self.net(x)

class DS(Dataset):
    def __init__(self, X, y): self.X=X.astype(np.float32); self.y=y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_model(model, X_tr, y_tr, epochs=200, lr=1e-3, wd=1e-4, patience=20, batch=64, cb=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    n = len(X_tr); vs = int(0.8*n)
    ds = DS(X_tr, y_tr)
    tdl = DataLoader(torch.utils.data.Subset(ds, range(vs)), batch_size=batch, shuffle=True)
    vdl = DataLoader(torch.utils.data.Subset(ds, range(vs, n)), batch_size=batch)
    best_val, wait, best_st = float('inf'), 0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tdl:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([loss_fn(model(xb), yb).item() for xb, yb in vdl])
        if v < best_val - 1e-12:
            best_val, wait = v, 0
            best_st = {k: v2.cpu().clone() for k,v2 in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
        if cb: cb(ep/epochs)
    if best_st: model.load_state_dict(best_st)
    return model

def do_predict(model, X, sy):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X.astype(np.float32))).numpy()
    return sy.inverse_transform(p)

def calc_metrics(pred, true):
    r = math.sqrt(mean_squared_error(true, pred))
    m = float(np.mean(np.abs((pred-true)/np.clip(np.abs(true),1e-6,None)))*100)
    r2 = r2_score(true, pred)
    return {"RMSE": r, "MAPE": m, "R²": r2}

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                color:#c8ff00;margin-bottom:0.2rem;'>StockMind</div>
    <div style='font-family:DM Mono,monospace;font-size:10px;color:#444;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1.5rem;'>
        Neural Forecasting Engine</div>
    """, unsafe_allow_html=True)

    st.markdown("**TICKER**")
    ticker = st.text_input("", value="NVDA", max_chars=6, label_visibility="collapsed").upper().strip()

    st.markdown("**HISTORY PERIOD**")
    period = st.selectbox("", ["1y","2y","5y","10y"], index=3, label_visibility="collapsed")

    st.markdown("**FORECAST HORIZON (days)**")
    horizon = st.slider("", 1, 30, 1, label_visibility="collapsed")

    st.markdown("**MODELS TO TRAIN**")
    do_base = st.checkbox("Baseline MLP  (64→1)", value=True)
    do_opt  = st.checkbox("Optimized MLP (64→32→1)", value=True)
    do_enh  = st.checkbox("Enhanced MLP  (128→64→1)", value=True)

    st.markdown("**FEATURES**")
    use_adv = st.checkbox("Advanced indicators (RSI, MACD, BB)", value=True)

    st.markdown("**TRAINING**")
    epochs   = st.slider("Max epochs", 50, 500, 200, step=50)
    patience = st.slider("Early-stop patience", 5, 50, 20)

    st.markdown("---")
    run_btn = st.button("▶  RUN EXPERIMENT")
    st.markdown("<div class='mono' style='margin-top:1rem;'>Data via yfinance · CPU training</div>",
                unsafe_allow_html=True)

# ── Hero ──
st.markdown(f"""
<div style='margin-bottom:1.5rem;'>
    <div class='hero-title'>Neural Stock<br><span class='hero-accent'>Forecasting</span></div>
    <div class='mono' style='margin-top:0.5rem;'>
        Ticker: <span style='color:#c8ff00'>{ticker}</span> &nbsp;·&nbsp;
        Period: <span style='color:#e8e6e0'>{period}</span> &nbsp;·&nbsp;
        Horizon: <span style='color:#e8e6e0'>{horizon}d</span> &nbsp;·&nbsp;
        Features: <span style='color:#e8e6e0'>{'Basic + Advanced' if use_adv else 'Basic only'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  Data & Features", "🧠  Train & Compare", "🔮  Predict"])

# ══ TAB 1: DATA ══
with tab1:
    try:
        with st.spinner("Fetching data..."):
            raw = download_stock(ticker, period)
        if raw.empty:
            st.error(f"No data for {ticker}.")
            st.stop()
        df = add_features(raw, use_advanced=use_adv)

        latest = float(df['Close'].iloc[-1]); prev = float(df['Close'].iloc[-2])
        chg = latest - prev; chg_pct = chg/prev*100
        hi52 = float(df['Close'].tail(252).max()); lo52 = float(df['Close'].tail(252).min())

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Latest Close", f"${latest:.2f}", f"{chg:+.2f} ({chg_pct:+.1f}%)")
        c2.metric("52W High", f"${hi52:.2f}")
        c3.metric("52W Low", f"${lo52:.2f}")
        c4.metric("Trading Days", f"{len(df):,}")

        st.markdown("---")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines',
                                  name='Close', line=dict(color='#c8ff00', width=1.5)))
        if use_adv:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_upper'], mode='lines',
                                      name='BB Upper', line=dict(color='#334455', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_lower'], mode='lines',
                                      name='BB Lower', line=dict(color='#334455', width=1, dash='dot'),
                                      fill='tonexty', fillcolor='rgba(30,60,80,0.08)'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'], mode='lines',
                                  name='MA 20', line=dict(color='#ff6b6b', width=1, dash='dash')))
        fig.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} · Price History", font=dict(size=13, color='#888')),
            height=320)
        st.plotly_chart(fig, use_container_width=True)

        if use_adv:
            cl, cr = st.columns(2)
            with cl:
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], mode='lines',
                                         name='RSI 14', line=dict(color='#a78bfa', width=1.5)))
                fr.add_hline(y=70, line_dash="dot", line_color="#ff6b6b", line_width=1)
                fr.add_hline(y=30, line_dash="dot", line_color="#6bffb8", line_width=1)
                fr.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="RSI 14", font=dict(size=12, color='#888')),
                    height=200, yaxis=dict(range=[0,100], **PLOTLY_LAYOUT['yaxis']))
                st.plotly_chart(fr, use_container_width=True)
            with cr:
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines',
                                         name='MACD', line=dict(color='#c8ff00', width=1.5)))
                fm.add_trace(go.Scatter(x=df['Date'], y=df['MACD_sig'], mode='lines',
                                         name='Signal', line=dict(color='#ff6b6b', width=1, dash='dash')))
                fm.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="MACD", font=dict(size=12, color='#888')), height=200)
                st.plotly_chart(fm, use_container_width=True)

        with st.expander("Feature matrix (last 10 rows)"):
            fcols = [c for c in df.columns if c not in ['Date','Open','High','Low','Volume','Adj Close']]
            st.dataframe(df[fcols].tail(10).style.format("{:.4f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ══ TAB 2: TRAIN ══
with tab2:
    if not run_btn:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0;'>
            <div style='font-family:DM Mono,monospace;font-size:13px;color:#333;
                        text-transform:uppercase;letter-spacing:0.15em;'>
                Configure parameters in the sidebar<br>then press ▶ RUN EXPERIMENT
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Preparing data..."):
                raw = download_stock(ticker, period)
                df  = add_features(raw, use_advanced=use_adv)
                base_f = [c for c in df.columns
                          if c in ['LogRet','VolChange'] or c.startswith('MA_') or c.startswith('STD_')]
                adv_f  = base_f + (['RSI_14','MACD','MACD_sig','BB_upper','BB_lower'] if use_adv else [])
                feat_cols = adv_f if use_adv else base_f

                X, y = make_supervised(df, feat_cols, horizon)
                split = int(0.8*len(X))
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]
                sx = StandardScaler().fit(X_tr); sy = StandardScaler().fit(y_tr)
                X_tr_s = sx.transform(X_tr); X_te_s = sx.transform(X_te)
                y_tr_s = sy.transform(y_tr)

            st.markdown(f"""
            <div class='mono' style='margin-bottom:1rem;'>
                Train: <span style='color:#e8e6e0'>{split:,}</span> &nbsp;·&nbsp;
                Test: <span style='color:#e8e6e0'>{len(X_te):,}</span> &nbsp;·&nbsp;
                Features: <span style='color:#e8e6e0'>{len(feat_cols)}</span> &nbsp;·&nbsp;
                Horizon: <span style='color:#c8ff00'>{horizon}d</span>
            </div>""", unsafe_allow_html=True)

            results = {}; all_preds = {}
            configs = [
                ("Baseline MLP",  do_base, lambda: BaselineMLP(X_tr_s.shape[1])),
                ("Optimized MLP", do_opt,  lambda: OptimizedMLP(X_tr_s.shape[1])),
                ("Enhanced MLP",  do_enh,  lambda: EnhancedMLP(X_tr_s.shape[1])),
            ]
            for name, enabled, build in configs:
                if not enabled: continue
                st.markdown(f"<div class='mono'>Training {name}...</div>", unsafe_allow_html=True)
                prog = st.progress(0)
                mdl = train_model(build(), X_tr_s, y_tr_s, epochs=epochs, patience=patience,
                                  cb=lambda v, p=prog: p.progress(v))
                prog.progress(1.0)
                preds = do_predict(mdl, X_te_s, sy)
                results[name] = calc_metrics(preds, y_te)
                all_preds[name] = preds.flatten()

            if not results:
                st.warning("Select at least one model.")
                st.stop()

            best = min(results, key=lambda k: results[k]['RMSE'])
            st.markdown("---")

            cols = st.columns(len(results))
            for i, (name, m) in enumerate(results.items()):
                cols[i].metric(
                    ("🏆 " if name==best else "") + name,
                    f"RMSE {m['RMSE']:.2f}",
                    f"MAPE {m['MAPE']:.1f}%  ·  R² {m['R²']:.3f}",
                    delta_color="off"
                )

            if "Baseline MLP" in results and len(results) > 1:
                br = results["Baseline MLP"]["RMSE"]
                st.markdown("---")
                ic = st.columns(len(results)-1); idx=0
                for name, m in results.items():
                    if name=="Baseline MLP": continue
                    imp = (br - m['RMSE'])/br*100
                    ic[idx].metric(f"{name} vs Baseline", f"{imp:+.1f}% RMSE",
                                   delta_color="normal" if imp>0 else "inverse")
                    idx += 1

            st.markdown("---")
            dates_test = df['Date'].iloc[split+horizon: split+horizon+len(y_te)].reset_index(drop=True)
            N = min(200, len(y_te))
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=dates_test[-N:], y=y_te.flatten()[-N:],
                                         mode='lines', name='Actual',
                                         line=dict(color='#e8e6e0', width=2)))
            colors = ['#c8ff00','#ff6b6b','#6bccff']
            for i, (name, pred) in enumerate(all_preds.items()):
                fig_fc.add_trace(go.Scatter(x=dates_test[-N:], y=pred[-N:], mode='lines',
                                             name=name, line=dict(color=colors[i%3], width=1.5,
                                             dash='solid' if name==best else 'dot')))
            fig_fc.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"Forecast vs Actual · Last {N} test days",
                           font=dict(size=13, color='#888')), height=380)
            st.plotly_chart(fig_fc, use_container_width=True)

            fig_bar = go.Figure()
            ns = list(results.keys())
            fig_bar.add_trace(go.Bar(
                name='RMSE', x=ns, y=[results[n]['RMSE'] for n in ns],
                marker_color=['#c8ff00' if n==best else '#2a2a4a' for n in ns],
                text=[f"{results[n]['RMSE']:.2f}" for n in ns],
                textposition='outside', textfont=dict(size=10, color='#888')
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT,
                title=dict(text="RMSE Comparison", font=dict(size=12, color='#888')),
                height=240, showlegend=False, bargap=0.4)
            st.plotly_chart(fig_bar, use_container_width=True)

            # store in session
            st.session_state.update({
                'results': results, 'best': best, 'feat_cols': feat_cols,
                'sx': sx, 'sy': sy, 'ticker': ticker, 'use_adv': use_adv,
                'horizon': horizon, 'epochs': epochs, 'patience': patience,
                'period': period
            })

        except Exception as e:
            st.error(f"Training error: {e}")
            raise e

# ══ TAB 3: PREDICT ══
with tab3:
    if 'results' not in st.session_state:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0;'>
            <div style='font-family:DM Mono,monospace;font-size:13px;color:#333;
                        text-transform:uppercase;letter-spacing:0.15em;'>
                Train models first in the 🧠 Train & Compare tab
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        ss = st.session_state
        t = ss['ticker']; hor = ss['horizon']; adv = ss['use_adv']
        feat_cols = ss['feat_cols']; sx = ss['sx']; sy = ss['sy']
        results = ss['results']; best = ss['best']

        with st.spinner("Computing forecast..."):
            raw_l = download_stock(t, ss['period'])
            df_l  = add_features(raw_l, use_advanced=adv)
            X_all, y_all = make_supervised(df_l, feat_cols, hor)
            split_l = int(0.8*len(X_all))
            X_tr_l = sx.transform(X_all[:split_l])
            y_tr_l = sy.transform(y_all[:split_l])

            latest_raw = df_l[feat_cols].iloc[-1].values.astype(float).reshape(1,-1)
            latest_s   = sx.transform(latest_raw)
            latest_close = float(df_l['Close'].iloc[-1])
            latest_date  = str(df_l['Date'].iloc[-1])[:10]

            d = len(feat_cols)
            builders = {
                "Baseline MLP":  lambda: BaselineMLP(d),
                "Optimized MLP": lambda: OptimizedMLP(d),
                "Enhanced MLP":  lambda: EnhancedMLP(d),
            }
            best_mdl = train_model(builders[best](), X_tr_l, y_tr_l,
                                   epochs=ss['epochs'], patience=ss['patience'])
            pred_s = best_mdl(torch.FloatTensor(latest_s.astype(np.float32)))
            pred_price = float(sy.inverse_transform(pred_s.detach().numpy())[0][0])

        direction = "↑" if pred_price > latest_close else "↓"
        dir_color = "#c8ff00" if pred_price > latest_close else "#ff6b6b"
        pct_chg   = (pred_price - latest_close)/latest_close*100
        best_rmse = results[best]['RMSE']

        st.markdown(f"""
        <div style='text-align:center;padding:2rem 0 1rem;'>
            <div class='mono' style='margin-bottom:0.5rem;'>{t} · Latest close ({latest_date})</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#555;margin-bottom:1.5rem;'>
                ${latest_close:.2f}</div>
            <div class='mono' style='margin-bottom:0.3rem;color:#444;
                                     text-transform:uppercase;letter-spacing:0.15em;'>
                {hor}-Day Forecast · {best}</div>
            <div style='font-family:Syne,sans-serif;font-size:3.5rem;font-weight:800;
                        letter-spacing:-0.02em;color:{dir_color};'>
                {direction} ${pred_price:.2f}</div>
            <div style='font-family:DM Mono,monospace;font-size:1rem;
                        color:{dir_color};margin-top:0.3rem;'>
                {pct_chg:+.2f}% expected move</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        lo = pred_price - best_rmse; hi = pred_price + best_rmse
        st.markdown(f"""
        <div style='text-align:center;'>
            <div class='mono' style='color:#444;margin-bottom:0.3rem;'>±1 RMSE uncertainty band</div>
            <div style='font-family:DM Mono,monospace;font-size:1.1rem;color:#888;'>
                ${lo:.2f} — ${hi:.2f}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='mono' style='color:#444;margin-bottom:1rem;'>All model forecasts</div>",
                    unsafe_allow_html=True)
        res_cols = st.columns(len(results))
        for i, (name, m) in enumerate(results.items()):
            mdl_i = train_model(builders[name](), X_tr_l, y_tr_l,
                                epochs=ss['epochs'], patience=ss['patience'])
            p = float(sy.inverse_transform(
                mdl_i(torch.FloatTensor(latest_s.astype(np.float32))).detach().numpy()
            )[0][0])
            pc = (p - latest_close)/latest_close*100
            res_cols[i].metric(("🏆 " if name==best else "") + name, f"${p:.2f}",
                               f"{pc:+.2f}%", delta_color="normal" if pc>=0 else "inverse")

        st.markdown("---")
        st.markdown("""
        <div class='mono' style='color:#2a2a3a;font-size:10px;text-align:center;padding:1rem 0;'>
            ⚠ For educational purposes only · Not financial advice · Past performance ≠ future results
        </div>""", unsafe_allow_html=True)