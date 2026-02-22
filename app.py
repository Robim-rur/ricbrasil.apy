import streamlit as st
import pandas as pd
import numpy as np

# Tenta importar as bibliotecas e avisa se houver erro de instalação
try:
    import yfinance as yf
    import pandas_ta as ta
except ImportError:
    st.error("Erro: Bibliotecas faltando no servidor. Verifique o requirements.txt")
    st.stop()

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(page_title="RICBRASIL ELITE - 2026", layout="wide")

# =====================================================
# LISTAS DE ATIVOS (Integradas para evitar erro de importação)
# =====================================================
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

ativos_scan = sorted(set(acoes_100 + bdrs_50 + etfs_fiis_24))

# =====================================================
# MOTOR DE BACKTEST - RIGOR MÁXIMO (ELITE)
# =====================================================
def calcular_estatistica_elite(df, sinais, stop_loss=0.04, stop_gain=0.08):
    resultados = []
    if not sinais: return 0, 0, 0
    
    for i in sinais:
        if i + 1 >= len(df): continue
        entrada = df["Close"].iloc[i]
        stop = entrada * (1 - stop_loss)
        alvo = entrada * (1 + stop_gain)
        
        for j in range(i + 1, min(i + 22, len(df))): 
            if df["Low"].iloc[j] <= stop:
                resultados.append(-stop_loss)
                break
            if df["High"].iloc[j] >= alvo:
                resultados.append(stop_gain)
                break
                
    if len(resultados) < 10: return 0, 0, len(resultados)
    
    win_rate = len([r for r in resultados if r > 0]) / len(resultados)
    expectativa = np.mean(resultados)
    return win_rate, expectativa, len(resultados)

# =====================================================
# ANALISADOR DE ATIVOS
# =====================================================
def analisar_ativo_elite(df_d, df_w, ticker):
    resultados = []
    
    # Cálculos Diários
    df_d.ta.adx(append=True)
    df_d["EMA50"] = ta.ema(df_d["Close"], length=50)
    
    if len(df_d) > 50:
        # Filtro de Tendência e Força (ADX > 20)
        if df_d["ADX_14"].iloc[-1] > 20 and df_d["Close"].iloc[-1] > df_d["EMA50"].iloc[-1]:
            c1, c2, c3 = df_d.iloc[-3], df_d.iloc[-2], df_d.iloc[-1]
            is_123 = c2["Low"] < c1["Low"] and c3["Low"] > c2["Low"]
            is_inside = c3["High"] <= c2["High"] and c3["Low"] >= c2["Low"]
            
            if is_123 or is_inside:
                sinais_hist = []
                for k in range(50, len(df_d)-5):
                    if df_d["Close"].iloc[k] > df_d["EMA50"].iloc[k]:
                        sinais_hist.append(k)
                
                wr, exp, n = calcular_estatistica_elite(df_d, sinais_hist)
                
                # RIGOR MÁXIMO: WR >= 65% e Expectativa Positiva Relevante
                if wr >= 0.65 and exp > 0.01:
                    resultados.append({"Ativo": ticker, "Setup": "Diário Elite", "WR": wr, "Exp": exp, "Trades": n})

    # Cálculos Semanais
    df_w["EMA21"] = ta.ema(df_w["Close"], length=21)
    df_w.ta.obv(append=True)
    
    if len(df_w) > 20:
        if df_w["Close"].iloc[-1] > df_w["EMA21"].iloc[-1] and df_w["OBV"].iloc[-1] > df_w["OBV"].rolling(10).mean().iloc[-1]:
            max_10 = df_w["High"].rolling(10).max().iloc[-2]
            if df_w["Close"].iloc[-1] > max_10:
                sinais_hist_w = []
                for k in range(50, len(df_w)-5):
                    if df_w["Close"].iloc[k] > df_w["High"].rolling(10).max().iloc[k-1]:
                        sinais_hist_w.append(k)
                
                wr, exp, n = calcular_estatistica_elite(df_w, sinais_hist_w)
                
                # RIGOR SEMANAL: WR >= 70%
                if wr >= 0.70 and exp > 0.02:
                    resultados.append({"Ativo": ticker, "Setup": "Semanal Elite", "WR": wr, "Exp": exp, "Trades": n})

    return resultados

# =====================================================
# INTERFACE
# =====================================================
st.title("🛡️ RICBRASIL ELITE - Scanner 2026")
st.markdown("---")

if st.button("🚀 EXECUTAR VARREDURA DE ALTA PRECISÃO"):
    res_final = []
    progress = st.progress(0)
    
    # Downloads (5 anos para Diário, 8 anos para Semanal para garantir o backtest)
    dados_d = yf.download(ativos_scan, period="5y", interval="1d", group_by="ticker", progress=False)
    dados_w = yf.download(ativos_scan, period="8y", interval="1wk", group_by="ticker", progress=False)

    for i, ativo in enumerate(ativos_scan):
        try:
            df_d = dados_d[ativo].dropna()
            df_w = dados_w[ativo].dropna()
            if not df_d.empty and not df_w.empty:
                items = analisar_ativo_elite(df_d, df_w, ativo.replace(".SA", ""))
                if items: res_final.extend(items)
        except: pass
        progress.progress((i + 1) / len(ativos_scan))

    if res_final:
        df_res = pd.DataFrame(res_final)
        df_res["WinRate %"] = (df_res["WR"] * 100).round(1)
        df_res["Exp. Matemática %"] = (df_res["Exp"] * 100).round(2)
        df_res = df_res.sort_values(by="Exp", ascending=False)
        st.success(f"Filtro aplicado! {len(df_res)} ativos de alta confiança encontrados.")
        st.dataframe(df_res[["Ativo", "Setup", "WinRate %", "Exp. Matemática %", "Trades"]], use_container_width=True)
    else:
        st.warning("Nenhum ativo atingiu o nível ELITE hoje.")
