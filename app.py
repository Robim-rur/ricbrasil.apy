import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import StochasticOscillator

st.set_page_config(layout="wide")

STOPS = {
    "ACAO": {"loss": 0.05, "gains": [0.07, 0.08]},
    "BDR_FII": {"loss": 0.04, "gains": [0.06]}
}

STOP_LOSS_CANDIDATOS = [0.03, 0.04, 0.05, 0.06]
STOP_GAIN_CANDIDATOS = [0.04, 0.05, 0.06, 0.07, 0.08]

MIN_TRADES = 20

# ------------------------
# Universo de Ativos (Inserido diretamente)
# ------------------------

def carregar_universo():
    acoes_100 = [
        "RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
        "BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
        "COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
        "ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
        "HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
        "MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
        "RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
        "UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA","AURE3.SA","BHIA3.SA","CASH3.SA",
        "CVCB3.SA","DIRR3.SA","ENAT3.SA","GMAT3.SA","IFCM3.SA","INTB3.SA","JHSF3.SA","KEPL3.SA","MOVI3.SA","ORVR3.SA",
        "PETZ3.SA","PLAS3.SA","POMO4.SA","POSI3.SA","RANI3.SA","RAPT4.SA","STBP3.SA","TEND3.SA","TUPY3.SA",
        "BRSR6.SA","CXSE3.SA"
    ]

    bdrs_50 = [
        "AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA","NVDC34.SA","MELI34.SA",
        "BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA","WMTB34.SA","NIKE34.SA",
        "ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA","HDCO34.SA","INTC34.SA",
        "JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA","PFIZ34.SA","PMIC34.SA",
        "QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA","VZUA34.SA",
        "ABTT34.SA","AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA"
    ]

    etfs_fiis_24 = [
        "BOVA11.SA","IVVB11.SA","SMAL11.SA","HASH11.SA","GOLD11.SA","GARE11.SA","HGLG11.SA","XPLG11.SA","VILG11.SA",
        "BRCO11.SA","BTLG11.SA","XPML11.SA","VISC11.SA","HSML11.SA","MALL11.SA","KNRI11.SA","JSRE11.SA","PVBI11.SA",
        "HGRE11.SA","MXRF11.SA","KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA",
        "DIVO11.SA","NDIV11.SA","SPUB11.SA"
    ]
    
    # Unindo todas as listas
    universo = acoes_100 + bdrs_50 + etfs_fiis_24
    return [x.strip().upper() for x in universo if x.strip()]


# ------------------------
# Dados
# ------------------------

def baixar_dados(ticker):
    df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
    df = df.dropna()
    return df


# ------------------------
# Indicadores
# ------------------------

def calcular_indicadores(df):
    df = df.copy()
    df["ema69"] = EMAIndicator(df["Close"], window=69).ema_indicator()
    
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["adx"] = adx.adx()
    df["diplus"] = adx.adx_pos()
    df["diminus"] = adx.adx_neg()

    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


def calcular_semanal(df):
    semanal = df.resample("W-FRI").last()
    semanal["ema69"] = EMAIndicator(semanal["Close"], window=69).ema_indicator()
    return semanal[["Close", "ema69"]]


# ------------------------
# Candle de entrada
# ------------------------

def candle_valido(df, i):
    if i == 0: return False
    o, c, h, l = df["Open"].iloc[i], df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
    prev_high = df["High"].iloc[i - 1]
    rng = h - l
    corpo = abs(c - o)
    if rng == 0 or c <= o or corpo / rng < 0.5 or c < (l + 0.66 * rng) or c <= prev_high:
        return False
    return True


# ------------------------
# Setup
# ------------------------

def localizar_setups(df, semanal):
    sinais = []
    for i in range(1, len(df)):
        data = df.index[i]
        sem = semanal.loc[semanal.index <= data]
        if sem.empty: continue
        sem_row = sem.iloc[-1]
        row = df.iloc[i]
        
        if np.isnan(row["ema69"]) or row["Close"] <= row["ema69"] or sem_row["Close"] <= sem_row["ema69"]:
            continue
        if row["diplus"] <= row["diminus"] or row["stoch_k"] <= row["stoch_d"]:
            continue
        if not candle_valido(df, i):
            continue
        sinais.append(i)
    return sinais


# ------------------------
# Simulação
# ------------------------

def simular_trade(df, i, stop_loss, stop_gain):
    entrada = df["Close"].iloc[i]
    stop, alvo = entrada * (1 - stop_loss), entrada * (1 + stop_gain)
    for j in range(i + 1, len(df)):
        low, high = df["Low"].iloc[j], df["High"].iloc[j]
        if low <= stop: return -stop_loss, j - i
        if high >= alvo: return stop_gain, j - i
    return None, None


def rodar_simulacao(df, sinais, stop_loss, stop_gain):
    resultados, duracoes = [], []
    for i in sinais:
        r, d = simular_trade(df, i, stop_loss, stop_gain)
        if r is not None:
            resultados.append(r)
            duracoes.append(d)
    if not resultados: return None
    wins = [x for x in resultados if x > 0]
    losses = [x for x in resultados if x < 0]
    winrate = len(wins) / len(resultados)
    payoff = abs(np.mean(wins)) / abs(np.mean(losses)) if wins and losses else np.nan
    return {
        "trades": len(resultados),
        "winrate": winrate,
        "payoff": payoff,
        "expectativa": np.mean(resultados),
        "duracao_media": np.mean(duracoes)
    }


def melhor_combinacao(df, sinais):
    melhor = None
    for sl in STOP_LOSS_CANDIDATOS:
        for sg in STOP_GAIN_CANDIDATOS:
            res = rodar_simulacao(df, sinais, sl, sg)
            if res and res["trades"] >= MIN_TRADES:
                if melhor is None or res["expectativa"] > melhor["expectativa"]:
                    melhor = {"stop_loss": sl, "stop_gain": sg, **res}
    return melhor


# ------------------------
# Interface
# ------------------------

st.title("Scanner automático – setup pessoal com filtro estatístico")

tipo_ativo = st.selectbox("Tipo de ativos do universo", ["ACAO", "BDR_FII"])

if st.button("Rodar varredura no universo"):
    ativos = carregar_universo()
    st.write(f"Ativos no universo: {len(ativos)}")
    resultados = []
    barra = st.progress(0)

    for idx, ticker in enumerate(ativos):
        try:
            df = baixar_dados(ticker)
            if df.empty or len(df) < 300: continue
            
            df = calcular_indicadores(df)
            semanal = calcular_semanal(df)
            sinais = localizar_setups(df, semanal)

            if len(sinais) >= MIN_TRADES:
                stops = STOPS[tipo_ativo]
                for gain in stops["gains"]:
                    res = rodar_simulacao(df, sinais, stops["loss"], gain)
                    if res and res["trades"] >= MIN_TRADES and res["expectativa"] > 0:
                        melhor = melhor_combinacao(df, sinais)
                        resultados.append({
                            "Ativo": ticker,
                            "Trades": res["trades"],
                            "Stop oficial": f"-{int(stops['loss']*100)}%",
                            "Gain oficial": f"+{int(gain*100)}%",
                            "Win rate (%)": round(res["winrate"] * 100, 2),
                            "Payoff": round(res["payoff"], 2) if not np.isnan(res["payoff"]) else None,
                            "Expectativa": round(res["expectativa"], 4),
                            "Duração média (dias)": round(res["duracao_media"], 1),
                            "Melhor stop estatístico": f"-{int(melhor['stop_loss']*100)}%" if melhor else None,
                            "Melhor gain estatístico": f"+{int(melhor['stop_gain']*100)}%" if melhor else None,
                            "Expectativa ótima": round(melhor["expectativa"], 4) if melhor else None
                        })
        except:
            pass
        barra.progress((idx + 1) / len(ativos))

    if resultados:
        df_final = pd.DataFrame(resultados).sort_values(by="Expectativa", ascending=False)
        st.subheader("Ativos que passaram nos filtros + estatística")
        st.dataframe(df_final, use_container_width=True)
    else:
        st.info("Nenhum ativo do universo passou nos filtros.")
