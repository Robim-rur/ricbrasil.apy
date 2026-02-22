import streamlit as st
import pandas as pd
import numpy as np

# Verificação de Bibliotecas
try:
    import yfinance as yf
    import pandas_ta as ta
except ImportError:
    st.error("Erro: Bibliotecas faltando no servidor. Verifique o requirements.txt")
    st.stop()

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(page_title="SETUP RICBRASIL - Regras Fixas", layout="wide")

# =====================================================
# LISTAS DE ATIVOS
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
bdrs_fii = [
    "AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA","NVDC34.SA","MELI34.SA",
    "BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA","WMTB34.SA","NIKE34.SA",
    "ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA","HDCO34.SA","INTC34.SA",
    "JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA","PFIZ34.SA","PMIC34.SA",
    "QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA","VZUA34.SA",
    "ABTT34.SA","AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA",
    "BOVA11.SA","IVVB11.SA","SMAL11.SA","HASH11.SA","GOLD11.SA","GARE11.SA","HGLG11.SA","XPLG11.SA","VILG11.SA",
    "BRCO11.SA","BTLG11.SA","XPML11.SA","VISC11.SA","HSML11.SA","MALL11.SA","KNRI11.SA","JSRE11.SA","PVBI11.SA",
    "HGRE11.SA","MXRF11.SA","KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA"
]

ativos_scan = sorted(set(acoes_100 + bdrs_fii))

# =====================================================
# MOTOR DE BACKTEST - REGRAS FIXAS DO USUÁRIO
# =====================================================
def calcular_estatistica_fixa(df, sinais, ticker):
    """Aplica 5/8% para Ações e 4/6% para BDR/FII"""
    resultados = []
    
    # Define regras por tipo de ativo
    is_bdr_fii = any(x in ticker for x in ["34", "11"])
    sl = 0.04 if is_bdr_fii else 0.05
    sg = 0.06 if is_bdr_fii else 0.08

    if not sinais: return 0, 0, 0, sl, sg
    
    for i in sinais:
        if i + 1 >= len(df): continue
        entrada = df["Close"].iloc[i]
        stop, alvo = entrada * (1 - sl), entrada * (1 + sg)
        
        for j in range(i + 1, min(i + 30, len(df))):
            if df["Low"].iloc[j] <= stop:
                resultados.append(-sl); break
            if df["High"].iloc[j] >= alvo:
                resultados.append(sg); break
                
    if len(resultados) < 4: return 0, 0, len(resultados), sl, sg
    
    win_rate = len([r for r in resultados if r > 0]) / len(resultados)
    expectativa = np.mean(resultados)
    return win_rate, expectativa, len(resultados), sl, sg

# =====================================================
# ANALISADOR
# =====================================================
def analisar_ativo(df_d, df_w, ticker):
    resultados = []
    
    # TENDÊNCIA (EMA 69)
    df_d["EMA69"] = ta.ema(df_d["Close"], length=69)
    if len(df_d) > 30 and df_d["Close"].iloc[-1] > df_d["EMA69"].iloc[-1]:
        
        c1, c2, c3 = df_d.iloc[-3], df_d.iloc[-2], df_d.iloc[-1]
        is_123 = c2["Low"] < c1["Low"] and c3["Low"] > c2["Low"]
        is_inside = c3["High"] <= c2["High"] and c3["Low"] >= c2["Low"]
        
        if is_123 or is_inside:
            sinais_hist = [k for k in range(30, len(df_d)-5) if df_d["Close"].iloc[k] > df_d["EMA69"].iloc[k]]
            wr, exp, n, sl_ativo, sg_ativo = calcular_estatistica_fixa(df_d, sinais_hist, ticker)
            
            ent = round(max(c2["High"], c3["High"]), 2)
            alvo_p = round(ent * (1 + sg_ativo), 2)
            stop_p = round(ent * (1 - sl_ativo), 2)
            
            # Ricardo Brasil: Só mostramos se a expectativa for positiva
            if exp > 0:
                resultados.append({
                    "Ativo": ticker.replace(".SA", ""),
                    "Setup": "Diário",
                    "WR %": round(wr*100, 1),
                    "Exp %": round(exp*100, 2),
                    "Sinais": n,
                    "Entrada (R$)": ent,
                    "Alvo Gain (R$)": alvo_p,
                    "Stop Loss (R$)": stop_p
                })

    return resultados

# =====================================================
# INTERFACE
# =====================================================
st.title("🛡️ SCANNER RICBRASIL - Regras Fixas")
st.sidebar.write("### Suas Regras Ativas:")
st.sidebar.info("**AÇÕES:** Loss 5% | Gain 8%\n\n**BDR/FII:** Loss 4% | Gain 6%")

if st.button("🔍 EXECUTAR SCANNER"):
    res_final = []
    progress = st.progress(0)
    
    # Baixando dados dos últimos 3 anos para estatística robusta
    dados_d = yf.download(ativos_scan, period="3y", interval="1d", group_by="ticker", progress=False)

    for i, ativo in enumerate(ativos_scan):
        try:
            df_d = dados_d[ativo].dropna()
            if not df_d.empty:
                # O setup semanal pode ser adicionado aqui depois, se desejar
                items = analisar_ativo(df_d, None, ativo)
                if items: res_final.extend(items)
        except: pass
        progress.progress((i + 1) / len(ativos_scan))

    if res_final:
        df_res = pd.DataFrame(res_final).sort_values(by="Exp %", ascending=False)
        st.success(f"Encontrados {len(df_res)} ativos que 'pagam a conta' com suas regras!")
        st.dataframe(df_res, use_container_width=True)
    else:
        st.warning("Nenhum ativo com sinal e estatística positiva para suas regras hoje.")
