import streamlit as st
import pandas as pd
import sqlite3
import time
import yaml
import matplotlib.pyplot as plt

# Konfiguracja strony
st.set_page_config(
    page_title="Mini Hedge Fund Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("🧙‍♂️ Mini Hedge Fund - Quant Dashboard")

# Funkcja do ładowania danych
def load_data(db_path='hedge_fund.db'):
    try:
        conn = sqlite3.connect(db_path)
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        return trades, signals
    except Exception as e:
        st.error(f"Błąd bazy danych: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Ładowanie konfigu dla kontekstu
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
trades_df, signals_df = load_data()

# Sidebar
st.sidebar.header("Ustawienia")
st.sidebar.json(config)
if st.sidebar.button("Odśwież dane"):
    st.experimental_rerun()

# Layout: Górne KPI
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_trades = len(trades_df)
live_trades = len(trades_df[trades_df['type'] == 'LIVE']) if not trades_df.empty else 0
paper_trades = len(trades_df[trades_df['type'] == 'PAPER']) if not trades_df.empty else 0

kpi1.metric("Wszystkie Transakcje", total_trades)
kpi2.metric("Paper Trades", paper_trades)
kpi3.metric("Live Trades", live_trades)

# Obliczanie PnL (szacunkowe, bo w bazie mamy tylko wejścia w tym prostym modelu)
# W pełnym systemie tabela trades powinna mieć entry_price i exit_price.
# Tutaj mamy prosty log otwarcia. Aby pokazać PnL, musielibyśmy mieć log zamknięcia.
# Zrobimy symulację na podstawie sygnałów lub prosty licznik.

# Sekcja: Ostatnie Sygnały
st.subheader("📡 Ostatnie Sygnały Strategii")
if not signals_df.empty:
    st.dataframe(signals_df.style.applymap(
        lambda x: 'color: green' if x == 'BUY' else ('color: red' if x == 'SELL' else ''),
        subset=['action']
    ))
else:
    st.info("Brak sygnałów w bazie.")

# Sekcja: Historia Transakcji
st.subheader("📜 Dziennik Transakcji")
if not trades_df.empty:
    st.dataframe(trades_df)
    
    # Wykres cen wejścia (tylko jako demo)
    st.line_chart(trades_df['price'])
else:
    st.info("Brak transakcji w bazie.")

# Auto-refresh
time.sleep(1)
# st.experimental_rerun() # Odkomentuj dla auto-odświeżania (może obciążać)
