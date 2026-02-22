import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import StochasticOscillator

st.set_page_config(layout="wide", page_title="SETUP RICBRASIL")

STOPS = {
    "ACAO": {"loss": 0.05, "gains": 0.08},
    "BDR_FII": {"loss": 0.04, "gains": 0.06}
}

MIN_TRADES = 5 

# ------------------------
# Universo de Ativos
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
        "HGRE11.SA","MXRF11.SA","KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"
    ]
    return acoes_100 + bdrs_50 + etfs_fiis_24

# ------------------------
# Processamento
# ------------------------

def baixar_dados(ticker):
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    return df.dropna()

def calcular_indicadores(df):
    df = df.copy()
    df["ema50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["diplus"] = adx.adx_pos()
    df["diminus"] = adx.adx_neg()
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    return df

def calcular_semanal(df):
    semanal = df.resample("W-FRI").last().copy()
    semanal["ema29_sem"] = EMAIndicator(semanal["Close"], window=29).ema_indicator()
    adx_sem = ADXIndicator(high=semanal["High"], low=semanal["Low"], close=semanal["Close"], window=14)
    semanal["diplus_sem"] = adx_sem.adx_pos()
    semanal["diminus_sem"] = adx_sem.adx_neg()
    return semanal.dropna()

def localizar_setups(df, semanal):
    sinais = []
    for i in range(1, len(df)):
        data = df.index[i]
        sem_data = semanal[semanal.index <= data]
        if sem_data.empty: continue
        
        row = df.iloc[i]
        sem_row = sem_data.iloc[-1]
        
        # Filtros de Tendência (Diária e Semanal)
        c1 = row["Close"] > row["ema50"]
        c2 = sem_row["Close"] > sem_row["ema29_sem"]
        
        # Filtro DI Semanal (Exigência D+ > D-)
        c3 = sem_row["diplus_sem"] > sem_row["diminus_sem"]
        
        # Filtro DI Diário
        c4 = row["diplus"] > row["diminus"]
        
        # Filtro Momento (Estocástico)
        c5 = row["stoch_k"] > 25
        
        # Gatilho: Apenas candle de alta (Removido filtro de "força" rígido)
        c6 = row["Close"] > row["Open"]
        
        if all([c1, c2, c3, c4, c5, c6]):
            sinais.append(i)
    return sinais

def rodar_simulacao(df, sinais, sl, sg):
    res = []
    for i in sinais:
        ent = df["Close"].iloc[i]
        stop, alvo = ent * (1-sl), ent * (1+sg)
        for j in range(i+1, len(df)):
            if df["Low"].iloc[j] <= stop:
                res.append(-sl); break
            if df["High"].iloc[j] >= alvo:
                res.append(sg); break
    if not res: return None
    return {"trades": len(res), "winrate": len([x for x in res if x > 0])/len(res), "expectativa": np.mean(res)}

# ------------------------
# Interface
# ------------------------

st.title("🛡️ SETUP RICBRASIL")
st.markdown("---")

tipo = st.selectbox("Grupo de Ativos", ["ACAO", "BDR_FII"])

if st.button("🚀 EXECUTAR SCANNER"):
    ativos = carregar_universo()
    progresso = st.progress(0)
    resultados = []
    status = st.empty()

    for idx, t in enumerate(ativos):
        status.text(f"Analisando: {t}")
        try:
            dados = baixar_dados(t)
            if len(dados) < 200: continue
            
            df_ind = calcular_indicadores(dados)
            df_sem = calcular_semanal(dados)
            sinais = localizar_setups(df_ind, df_sem)
            
            if len(sinais) >= MIN_TRADES:
                conf = STOPS[tipo]
                backtest = rodar_simulacao(df_ind, sinais, conf["loss"], conf["gains"])
                
                if backtest and backtest["expectativa"] > 0:
                    resultados.append({
                        "Ativo": t,
                        "Sinais (5 anos)": backtest["trades"],
                        "Win Rate": f"{round(backtest['winrate']*100,1)}%",
                        "Expectativa": f"{round(backtest['expectativa']*100, 2)}%"
                    })
        except: pass
        progresso.progress((idx+1)/len(ativos))

    status.empty()
    if resultados:
        st.success(f"Sucesso! {len(resultados)} ativos encontrados.")
        st.dataframe(pd.DataFrame(resultados).sort_values("Expectativa", ascending=False), use_container_width=True)
    else:
        st.warning("Nenhum ativo atendeu ao critério DI+ > DI- Semanal com as médias atuais.")





