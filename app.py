import streamlit as st
import pandas as pd
import numpy as np

# Tenta importar as bibliotecas e avisa se houver erro de instalação
try:
    import yfinance as yf
    import pandas_ta as ta
except ImportError:
    st.error("Erro: Bibliotecas faltando. Verifique o requirements.txt")
    st.stop()

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(page_title="RICBRASIL ELITE - 2026", layout="wide")

# (Listas de ativos permanecem as mesmas - omitidas para brevidade, mas devem estar no seu código)
from ativos import ativos_scan # Ou cole suas listas aqui como antes

# =====================================================
# MOTOR DE BACKTEST - RIGOR MÁXIMO
# =====================================================
def calcular_estatistica_elite(df, sinais, stop_loss=0.04, stop_gain=0.08):
    """
    Rigor Ricardo Brasil: Payoff 2:1
    Exige amostragem e consistência.
    """
    resultados = []
    if not sinais: return 0, 0, 0
    
    for i in sinais:
        if i + 1 >= len(df): continue
        entrada = df["Close"].iloc[i]
        stop = entrada * (1 - stop_loss)
        alvo = entrada * (1 + stop_gain)
        
        # Simulação de saída
        for j in range(i + 1, min(i + 22, len(df))): # Limite de 1 mês para o trade
            if df["Low"].iloc[j] <= stop:
                resultados.append(-stop_loss)
                break
            if df["High"].iloc[j] >= alvo:
                resultados.append(stop_gain)
                break
                
    # RIGOR: Mínimo de 10 trades históricos para validar
    if len(resultados) < 10: return 0, 0, len(resultados)
    
    win_rate = len([r for r in resultados if r > 0]) / len(resultados)
    expectativa = np.mean(resultados)
    
    return win_rate, expectativa, len(resultados)

# =====================================================
# ANALISADORES DE SETUPS COM FILTROS DE TENDÊNCIA 2026
# =====================================================
def analisar_ativo_elite(df_d, df_w, ticker):
    resultados = []
    
    # Adicionando ADX para medir força da tendência
    df_d.ta.adx(append=True)
    df_d["EMA50"] = ta.ema(df_d["Close"], length=50) # EMA 50 é o suporte institucional em 2026
    
    # 1. Setup Diário (123 / Inside) - RIGOR ADICIONAL
    if len(df_d) > 50:
        # Só opera se ADX > 20 (Tendência real)
        if df_d["ADX_14"].iloc[-1] > 20 and df_d["Close"].iloc[-1] > df_d["EMA50"].iloc[-1]:
            
            c1, c2, c3 = df_d.iloc[-3], df_d.iloc[-2], df_d.iloc[-1]
            is_123 = c2["Low"] < c1["Low"] and c3["Low"] > c2["Low"]
            is_inside = c3["High"] <= c2["High"] and c3["Low"] >= c2["Low"]
            
            if is_123 or is_inside:
                sinais_hist = []
                for k in range(50, len(df_d)-5):
                    # Filtro histórico simplificado para o backtest
                    if df_d["Close"].iloc[k] > df_d["EMA50"].iloc[k]:
                        sinais_hist.append(k)
                
                wr, exp, n = calcular_estatistica_elite(df_d, sinais_hist)
                
                # RIGOR RICBRASIL: WR > 65% e Expectativa robusta
                if wr >= 0.65 and exp > 0.01:
                    resultados.append({
                        "Ativo": ticker, 
                        "Setup": "Diário Elite", 
                        "WinRate": wr, 
                        "Expectativa": exp, 
                        "Trades": n
                    })

    # 2. Setup Semanal (OBV + EMA21)
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
                
                # RIGOR SEMANAL: WR > 70% (Position exige mais acerto)
                if wr >= 0.70 and exp > 0.02:
                    resultados.append({
                        "Ativo": ticker, 
                        "Setup": "Semanal Elite", 
                        "WinRate": wr, 
                        "Expectativa": exp, 
                        "Trades": n
                    })

    return resultados

# =====================================================
# INTERFACE
# =====================================================
st.title("🛡️ RICBRASIL ELITE - Scanner 2026")
st.markdown("---")
st.sidebar.header("Parâmetros de Rigor")
st.sidebar.write("- Backtest: 5 Anos")
st.sidebar.write("- Win Rate Mínimo: 65%")
st.sidebar.write("- Amostragem Mínima: 10 trades")

if st.button("🚀 EXECUTAR VARREDURA DE ALTA PRECISÃO"):
    res_final = []
    progress = st.progress(0)
    
    # Downloads (Aumentado para 5 anos para o rigor do backtest)
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
        df_res["WinRate (%)"] = (df_res["WinRate"] * 100).round(1)
        df_res["Exp. Matemática (%)"] = (df_res["Expectativa"] * 100).round(2)
        df_res = df_res.sort_values(by="Expectativa", ascending=False)
        
        st.success(f"Filtro aplicado! Restaram apenas {len(df_res)} ativos de alta confiança.")
        st.dataframe(df_res[["Ativo", "Setup", "WinRate (%)", "Exp. Matemática (%)", "Trades"]], use_container_width=True)
    else:
        st.warning("Nenhum ativo atingiu o nível ELITE de probabilidade hoje.")

