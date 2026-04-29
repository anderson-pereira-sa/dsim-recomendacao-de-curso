import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Inteligência de Mercado",
    page_icon="📊",
    layout="wide")

st.title("Modelo Recomendação de Curso")
st.caption("Machine Learning aplicado à análise de risco e oportunidade em matrículas")

# =========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pickle.load(open("dataset.pkl", "rb"))

df_matricula = load_data()

# ==========================================================
# RÓTULOS DE EXIBIÇÃO (UI)
# ==========================================================

labels_exibicao = {
    'QTD_CONC': 'CONCORRÊNCIA)',
    # 'QTD_MAT_CONC': 'MATRÍCULA CONC. (INEP)',
    'QTD_EMPRESAS': 'EMPRESAS (RAIS)',
    # 'QTD_VINCULOS': 'VÍNCULOS (RAIS)',
    'SALARIO_MEDIO': 'SALÁRIO MÉDIO (CAGED)',
    'SALDO_EMPREGO': 'SALDO EMPREGO (CAGED)',
    'VLR_MEDIO_BENEFICIO': 'VALOR MÉDIO BOLSA FAMÍLIA',
    'MAT_PAG': 'MATR.(SENAI)',
    'FAIXA_MAT': 'FAIXA DE MATR. (SENAI)'
}

# ==========================================================
# LOAD MODELO + ARTEFATOS (DO data_model.py)
# ==========================================================

@st.cache_resource
def load_model_artifacts():

    model = XGBClassifier()
    model.load_model("modelo_xgbc.json")

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open("categoricas.pkl", "rb") as f:
        categoricas = pickle.load(f)

    with open("numericas.pkl", "rb") as f:
        numericas = pickle.load(f)

    with open("feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)

    return model, encoder, le, categoricas, numericas, feature_order


modelo, encoder, le, categoricas_ohencoder, numericas_ohencoder, feature_order = load_model_artifacts()
FAIXAS = list(le.classes_)

# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def build_X(row):
    row = row.copy(deep=True)
    for col in numericas_ohencoder:
        if col not in row.index:
            row[col] = 0.0

    X_cat = encoder.transform(row[categoricas_ohencoder].to_frame().T)
    X_num = row[numericas_ohencoder].astype(float).values.reshape(1, -1)

    return np.hstack([X_num, X_cat]).astype(float)


def impacto_variaveis_locais(linha_real,linha_sim, faixa_idx, delta_padrao=0.10, epsilon=1.0):
    impactos = {}

    X_base = build_X(linha_sim)
    prob_base = modelo.predict_proba(X_base)[0][faixa_idx]

    for var in numericas_ohencoder:
        if var not in linha_sim.index:
            continue

        linha_temp = linha_sim.copy(deep=True)
        valor_sim = linha_sim[var]

        if valor_sim > 0:
            linha_temp[var] = valor_sim * (1 + delta_padrao)
        else:
            linha_temp[var] = epsilon

        X_temp = build_X(linha_temp)
        prob_temp = modelo.predict_proba(X_temp)[0][faixa_idx]
        impactos[var] = prob_temp - prob_base

    return impactos

def grafico_real_barras_horizontais(probs_atual, ano_atual):
    df_bar = pd.DataFrame({"Faixa": FAIXAS, "Probabilidade": probs_atual})
    cores = {
        FAIXAS[0]: "#1D3691",  # Abaixo de 21 matrículas
        FAIXAS[1]: "#795821",  # Entre 21 e 40
        FAIXAS[2]: "#195E32"   # Acima de 40 matrículas
    }

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_bar["Probabilidade"],
            y=df_bar["Faixa"],
            orientation="h",
            marker_color=[cores[f] for f in df_bar["Faixa"]],
            text=[f"{p*100:.1f}%" for p in df_bar["Probabilidade"]],
            textposition="auto",
            textfont=dict(
                size=18
                # color="black"   # ✅ Garante contraste executivo
            )))

    fig.update_layout(
        title=f"Distribuição das Probabilidades – {ano_atual}",
        xaxis_title="Probabilidade",
        yaxis_title="Faixa de Matrícula",
        xaxis_tickformat=".0%",
        template="plotly_white",
        height=350,
        margin=dict(l=140, r=40, t=60, b=40))
    return fig

def probabilidades_por_ano(df_base, unidade, curso):
    resultados = []
    df_base_contexto = (
        df_base[
            (df_base['UNIDADE'] == unidade) &
            ((curso == 'GLOBAL') | (df_base['CURSO'] == curso))]
        .groupby(['ANO', 'UNIDADE', 'CURSO'], as_index=False)
        .agg(
            QTD_CONC=('QTD_CONC', 'max'),
            QTD_EMPRESAS=('QTD_EMPRESAS', 'max'),
            SALARIO_MEDIO=('SALARIO_MEDIO', 'mean'),
            SALDO_EMPREGO=('SALDO_EMPREGO', 'sum'),
            MAT_PAG=('MAT_PAG', 'sum')).sort_values('ANO'))
    
    if df_base_contexto.empty:
        return pd.DataFrame(columns=['ANO', *FAIXAS])

    linha_base = df_base_contexto.iloc[-1].copy(deep=True)

    for ano in sorted(df_base['ANO'].unique()):
        linha_temp = linha_base.copy(deep=True)
        linha_temp['ANO'] = ano

        probs = modelo.predict_proba(build_X(linha_temp))[0]

        linha_resultado = {'ANO': ano}
        for faixa, p in zip(FAIXAS, probs):
            linha_resultado[faixa] = p

        resultados.append(linha_resultado)

    return pd.DataFrame(resultados)

def grafico_real_linhas(df_series):
    cores = {
        FAIXAS[0]: "#1D3691",  # Abaixo de 21 matrículas
        FAIXAS[1]: "#795821",  # Entre 21 e 40
        FAIXAS[2]: "#195E32"   # Acima de 40 matrículas
    }

    fig = go.Figure()

    for faixa in FAIXAS:
        fig.add_trace(
            go.Scatter(
                x=df_series["ANO"],
                y=df_series[faixa],
                mode="lines+markers",
                name=faixa,
                line=dict(
                    color=cores[faixa],
                    width=3,
                    shape="spline"),  # ✅ suavização visual
                marker=dict(size=6)))

    fig.update_layout(
        title="Evolução Histórica das Probabilidades por Faixa",
        xaxis_title="Ano",
        yaxis_title="Probabilidade",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=360,
        legend_title_text="Faixa de Matrícula")

    return fig

def analise_executiva_prob_real(
    faixa_dominante,
    prob_dominante,
    impactos_dict,
    limite_otimo=0.69,
    limite_relevante_pp=1.0
):
    impactos = pd.Series(impactos_dict).sort_values(key=abs, ascending=False)
    impactos_neg = impactos[impactos < -limite_relevante_pp / 100]

    def nomes(vars):
        return [labels_exibicao.get(v, v) for v in vars]

    prob_pct = prob_dominante * 100

    # ======================================================
    # FAIXA BAIXA
    # ======================================================
    if faixa_dominante == "Abaixo de 21 matrículas":
        texto = (
            f"O cenário atual apresenta **baixa demanda**, com "
            f"**{prob_pct:.1f}%** de probabilidade concentrada na faixa "
            f"**abaixo de 21 matrículas**. "
            "Esse patamar indica um **contexto desfavorável**, que tende a se manter "
            "na ausência de mudanças estruturais. "
            "A recomendação é utilizar o **simulador** para avaliar quais variáveis "
            "podem contribuir para a reversão desse cenário."
        )
        return texto, "warning"

    # ======================================================
    # FAIXA INTERMEDIÁRIA
    # ======================================================
    if faixa_dominante == "Entre 21 e 40":
        texto = (
            f"O cenário atual indica **demanda intermediária**, com "
            f"**{prob_pct:.1f}%** de probabilidade na faixa "
            f"**entre 21 e 40 matrículas**. "
            "Esse resultado sugere **equilíbrio**, ainda com espaço para evolução. "
            "A recomendação é monitorar os fatores de mercado e, se necessário, "
            "avaliar ajustes por meio do **simulador**."
        )
        return texto, "info"

    # ======================================================
    # FAIXA ALTA
    # ======================================================
    if faixa_dominante == "Acima de 40" and prob_dominante >= limite_otimo:

        if not impactos_neg.empty:
            texto = (
                f"O cenário atual é **altamente favorável**, com "
                f"**{prob_pct:.1f}%** de probabilidade na faixa "
                f"**acima de 40 matrículas**. "
                "O modelo indica, contudo, que variáveis como "
                f"**{', '.join(nomes(impactos_neg.index[:2]))}** "
                "representam **fatores de atenção**, pois podem provocar "
                "deslocamentos marginais do perfil de demanda. "
                "A recomendação é **manter a estratégia atual** e "
                "**monitorar a evolução desses indicadores**."
            )
        else:
            texto = (
                f"O cenário atual é **altamente favorável**, com "
                f"**{prob_pct:.1f}%** de probabilidade na faixa "
                f"**acima de 40 matrículas**. "
                "Não há sinais relevantes de risco no curto prazo. "
                "A recomendação é **manter a estratégia vigente**."
            )

        return texto, "success"

    # ======================================================
    # FALLBACK
    # ======================================================
    return (
        "O cenário atual apresenta comportamento equilibrado. "
        "Recomenda-se acompanhamento contínuo.",
        "info"
    )


def analise_executiva_cenario_simulado(faixa_simulada,impactos_dict,limite_relevante_pp=1.0):
    impactos = pd.Series(impactos_dict).sort_values(key=abs, ascending=False)
    impactos_relevantes = impactos[abs(impactos) > limite_relevante_pp / 100]

    def nomes(vars):
        return [labels_exibicao.get(v, v) for v in vars]

    if impactos_relevantes.empty:
        return (
            "As alterações realizadas não provocaram mudanças relevantes "
            "no cenário projetado. O perfil de demanda permanece estável.",
            "info")

    texto = (
        f"No cenário simulado, a faixa dominante passa a ser "
        f"**{faixa_simulada}**. "
        "O modelo indica que as mudanças realizadas afetam principalmente "
        f"**{', '.join(nomes(impactos_relevantes.index[:2]))}**, "
        "resultando em **deslocamento do perfil de demanda** em relação ao cenário atual.")
    return texto, "info"


def pizza_sim(values, title):
    fig = go.Figure(
        go.Pie(
            labels=FAIXAS,
            values=np.nan_to_num(values),
            hole=0.55,
            textinfo='percent',
            texttemplate='%{percent:.1%}',
            textfont=dict(size=18),
            marker=dict(
                colors=["#1D3691", "#795821", "#195E32"],
                line=dict(color='white', width=2)
            )
        )
    )
    fig.update_layout(height=460, title=title, template="plotly_white")
    return fig


def gerar_recomendacao(faixa_dominante, prob_max):
    if faixa_dominante == "Acima de 40" and prob_max >= 0.70:
        return "✅ Recomendar fortemente a oferta"

    if faixa_dominante == "Acima de 40" and prob_max >= 0.50:
        return "🟢 Recomendar a oferta"

    if faixa_dominante == "Entre 21 e 40" and prob_max >= 0.50:
        return "🟡 Oferta com cautela"

    if faixa_dominante == "Abaixo de 21" and prob_max >= 0.60:
        return "🔴 Não priorizar oferta"

    return "⚠️ Analisar manualmente"

def gerar_cenario_futuro(linha_base, ano_futuro, cenario="base"):
    linha = linha_base.copy(deep=True)
    linha['ANO'] = ano_futuro

    if cenario == "conservador":
        linha['SALDO_EMPREGO'] *= 1.00
        linha['QTD_CONC'] *= 1.05
        linha['SALARIO_MEDIO'] *= 1.02
        linha['VLR_MEDIO_BENEFICIO'] *= 1.02

    elif cenario == "otimista":
        linha['SALDO_EMPREGO'] *= 1.10
        linha['QTD_CONC'] *= 0.95
        linha['SALARIO_MEDIO'] *= 1.05
        linha['VLR_MEDIO_BENEFICIO'] *= 1.03

    else:  # cenário base
        linha['SALDO_EMPREGO'] *= 0.95
        linha['QTD_CONC'] *= 1.05
        linha['SALARIO_MEDIO'] *= 1.03
        linha['VLR_MEDIO_BENEFICIO'] *= 1.02

    return linha


def gerar_matriz_curso_unidade_futuro(df_historico, ano_futuro, cenario="base"):
    resultados = []
    unidades = df_historico['UNIDADE'].unique()
    cursos = df_historico['CURSO'].unique()

    for unidade in unidades:

        # -------------------------------
        # Contexto local da unidade
        # -------------------------------
        df_unid = df_historico[df_historico['UNIDADE'] == unidade]

        base_unidade = (
            df_unid
            .sort_values('ANO')
            .iloc[-1]
            .copy(deep=True)
        )

        for curso in cursos:

            df_uc = df_historico[
                (df_historico['UNIDADE'] == unidade) &
                (df_historico['CURSO'] == curso)
            ]

            # ======================================================
            # CASO 1 — Curso já existiu na unidade
            # ======================================================
            if not df_uc.empty:
                linha_base = (
                    df_uc
                    .sort_values('ANO')
                    .iloc[-1]
                    .copy(deep=True)
                )
                historico = True

            # ======================================================
            # CASO 2 — Curso nunca existiu na unidade
            # ======================================================
            else:
                df_curso_global = df_historico[
                    df_historico['CURSO'] == curso
                ]

                if df_curso_global.empty:
                    continue  # curso sem histórico nenhum

                linha_base = base_unidade.copy(deep=True)
                linha_base['CURSO'] = curso

                # imputação defensável
                linha_base['QTD_CONC'] = df_curso_global['QTD_CONC'].median()
                historico = False

            # ======================================================
            # CENÁRIO FUTURO
            # ======================================================
            linha_futura = gerar_cenario_futuro(
                linha_base,
                ano_futuro=ano_futuro,
                cenario=cenario
            )

            X = build_X(linha_futura)
            probs = modelo.predict_proba(X)[0]

            faixa_idx = int(np.argmax(probs))
            faixa_nome = FAIXAS[faixa_idx]
            prob_max = float(np.max(probs))

            recomendacao = formatar_recomendacao_celula(
                faixa_dominante=faixa_nome,
                prob_max=prob_max
            )


            resultados.append({
                'ANO_SIM': ano_futuro,
                'CENARIO': cenario,
                'UNIDADE': unidade,
                'CURSO': curso,
                'FAIXA_DOMINANTE': faixa_nome,
                'PROB_MAX': prob_max,
                'col_recom': recomendacao,
                'HISTORICO_NA_UNIDADE': historico
            })

    return pd.DataFrame(resultados)

def formatar_recomendacao_celula(faixa_dominante, prob_max):
    prob_pct = round(prob_max * 100, 1)
    if faixa_dominante == "Acima de 40" and prob_max >= 0.70:
        icone = "✅"
    elif faixa_dominante == "Acima de 40":
        icone = "🟢"
    elif faixa_dominante == "Entre 21 e 40":
        icone = "🟡"
    elif faixa_dominante == "Abaixo de 21":
        icone = "🔴"
    else:
        icone = "⚠️"

    return f"{icone} {faixa_dominante} ({prob_pct}%)"

# ==========================================================
# ABAS
# ==========================================================

tab1, tab2 = st.tabs(["📌 Contexto do Modelo", " Recomendação CHP"])

# ==========================================================
# ABA 1
# ==========================================================

with tab1:

    st.subheader("📌 Sobre o Modelo")

    st.markdown("""
    ⁂ Este modelo utiliza técnicas de machine learning para 
    apoiar decisões estratégicas relacionadas à recomendação de curso por matrículas:

       **📍Abaixo de 21 matrículas |📍 Entre 21 e 40 matrículas |📍 Acima de 40 matrículas**

    ⁂ O foco do modelo é **apoio à decisão**, e **não previsão** pontual.
                
    ⁂ Este modelo responde diversas perguntas como, por exemplo:
                
        ✓ "Qual é a probabilidade de uma unidade estar em cada faixa de matrícula?
        ✓ “Qual o risco/oportunidade dado que já têm as previsões de mercado (de empresas, vínculos, salário médio ou saldo de emprego)?”
        ✓ “Quais variáveis realmente influenciam nas probabilidades?”
    """)

# ==========================================================
# ABA 2 – PAINEL CENTRAL
# ==========================================================

with tab2:

    st.subheader("📌 Painel de Análise")

    # ==========================================================
    # SIDEBAR – FILTROS DE CONTEXTO (COM GLOBAL)
    # ==========================================================

    st.sidebar.header("🔷 Contexto da Análise")

    # ANO
    ano_max = int(df_matricula['ANO'].max())
    ano_sel = st.sidebar.selectbox(
        "ANO "
        "(Apenas para o Contexto Histórico)", 
        ['TODOS'] + sorted(df_matricula['ANO'].unique()))

    ano_modelo = ano_max if ano_sel == 'TODOS' else ano_sel
    df_ano = df_matricula if ano_sel == 'TODOS' else df_matricula[df_matricula['ANO'] == ano_sel]

    # UNIDADE
    unidade_sel = st.sidebar.selectbox("UNIDADE", sorted(df_ano['UNIDADE'].unique()))
    df_unidade = df_ano[df_ano['UNIDADE'] == unidade_sel]

    # CURSO
    curso_sel = st.sidebar.selectbox("CURSO", ['GLOBAL'] + sorted(df_unidade['CURSO'].unique()))

    df_base = (
        df_matricula[df_matricula['UNIDADE'] == unidade_sel]
        if curso_sel == 'GLOBAL'
        else df_matricula[(df_matricula['UNIDADE'] == unidade_sel) & (df_matricula['CURSO'] == curso_sel)]
    )

    linha_real = df_base[df_base['ANO'] == df_base['ANO'].max()].iloc[-1].copy()
 
    # ---------- PROBABILIDADE REAL ----------
    probs_real = modelo.predict_proba(build_X(linha_real))[0]

    # ---------- IMPACTOS NO CENÁRIO REAL ----------
    faixa_idx_real = int(np.argmax(probs_real))

    # Para o cenário real, usamos linha_real como baseline
    impactos_real = impacto_variaveis_locais(
        linha_real=linha_real,
        linha_sim=linha_real, 
        faixa_idx=faixa_idx_real)

    # ==========================================================
    # LAYOUT: REAL | DIVISOR | SIMULADO
    # ==========================================================

    # col_real, col_div, col_sim , col_recom = st.columns([1.8, 0.05, 1.2, 1.4])
    col_real, col_div, col_sim = st.columns([1.8, 0.05, 1.2])

    # ---------- REAL ----------
    with col_real:

        st.subheader("📊 Cenário REAL")

        g_bar, texto = st.columns(2)

        with g_bar:
            st.plotly_chart(
                grafico_real_barras_horizontais(
                    probs_atual=probs_real,
                    ano_atual=ano_modelo
                ),
                use_container_width=True
            )

        with texto:
            st.markdown("#### ✦ Probabilidade REAL")
            faixa_dominante = FAIXAS[int(np.argmax(probs_real))]
            prob_dominante = float(np.max(probs_real))
            
            texto_prob_real, tipo_prob_real = analise_executiva_prob_real(
                faixa_dominante=faixa_dominante,
                prob_dominante=prob_dominante,
                impactos_dict=impactos_real
            )

            if tipo_prob_real == "warning":
                st.warning(texto_prob_real)
            elif tipo_prob_real == "info":
                st.info(texto_prob_real)
            else:
                st.success(texto_prob_real)

        # ---------- HISTÓRICO (INDEPENDENTE DO FILTRO DE ANO) ----------
        df_series = probabilidades_por_ano(
            df_base=df_matricula,
            unidade=unidade_sel,
            curso=curso_sel
            # municipio=municipio_sel
        )

        st.plotly_chart(grafico_real_linhas(df_series), use_container_width=True)

        # ---------- TABELA HISTÓRICA ----------
        st.subheader("📋 Contexto do Município Selecionado")

        # ---------- TABELA HISTÓRICA ----------
        if ano_sel == 'TODOS':
            df_contexto_historico = df_matricula[
                (df_matricula['UNIDADE'] == unidade_sel) &
                ((curso_sel == 'GLOBAL') | (df_matricula['CURSO'] == curso_sel))
            ]
        else:
            df_contexto_historico = df_matricula[
                (df_matricula['UNIDADE'] == unidade_sel) &
                (df_matricula['ANO'] == ano_sel) &
                ((curso_sel == 'GLOBAL') | (df_matricula['CURSO'] == curso_sel))
            ]

        df_contexto_historico = df_contexto_historico.sort_values('ANO')
        df_contexto_historico['SALARIO_MEDIO'] = round(df_contexto_historico['SALARIO_MEDIO'], 2)
        df_contexto_historico = df_contexto_historico[[ 'ANO', 'UNIDADE', 'CURSO', 'FAIXA_MAT', 'MAT_PAG', 'QTD_CONC', 'QTD_EMPRESAS', 'SALDO_EMPREGO','SALARIO_MEDIO' ]]
        df_contexto_historico = df_contexto_historico.rename(columns=labels_exibicao)

        st.dataframe(
            df_contexto_historico,
            use_container_width=True,
            hide_index=True
        )

    # ---------- DIVISOR ----------
    with col_div:
        if curso_sel == 'GLOBAL':
            for _ in range(30):
                st.markdown("|")
        else:
            for _ in range(24):
                st.markdown("|")

    # ---------- SIMULADO ----------
    with col_sim:

        st.subheader("🧪 Cenário SIMULADO")
        st.markdown("##### Parâmetros do cenário projetado")

        # c1, c2, c3 = st.columns(3)
        c1, c2 = st.columns(2)

        with c1:
            conc_sim = st.number_input(labels_exibicao['QTD_CONC'], value=int(linha_real['QTD_CONC']))
            emp_sim = st.number_input(labels_exibicao['QTD_EMPRESAS'], value=int(linha_real['QTD_EMPRESAS']))
            bf_sim = st.number_input(labels_exibicao['VLR_MEDIO_BENEFICIO'], value=float(linha_real['VLR_MEDIO_BENEFICIO']))

        with c2:
            sal_sim = st.number_input(labels_exibicao['SALARIO_MEDIO'], value=float(linha_real['SALARIO_MEDIO']))
            saldo_sim = st.number_input(labels_exibicao['SALDO_EMPREGO'], value=int(linha_real['SALDO_EMPREGO']))

        linha_sim = linha_real.copy(deep=True)
        valores_simulados = {
            'QTD_EMPRESAS': emp_sim,
            # 'QTD_VINCULOS': vinc_sim,
            'VLR_MEDIO_BENEFICIO': bf_sim,
            'QTD_CONC': conc_sim,
            'SALARIO_MEDIO': sal_sim,
            'SALDO_EMPREGO': saldo_sim
            }

        for col, val in valores_simulados.items():
            linha_sim[col] = val

        probs_sim = modelo.predict_proba(build_X(linha_sim))[0]

        faixa_idx = int(np.argmax(probs_sim))
        faixa_nome = FAIXAS[faixa_idx]

        impactos = impacto_variaveis_locais(linha_real, linha_sim, faixa_idx)
        texto_exec, _ = analise_executiva_cenario_simulado(faixa_simulada=faixa_nome, impactos_dict=impactos)

        st.plotly_chart(pizza_sim(probs_sim, "Distribuição SIMULADA"), True)

        var_sim, an_sim = st.columns(2)

        with var_sim:
            st.markdown("#### ✦ Variáveis mais influentes")
            for v, i in pd.Series(impactos).sort_values(key=abs, ascending=False).head(5).items():
                st.write(f"▶ **{labels_exibicao.get(v, v)}** ({i*100:+.1f} p.p.)")

        with an_sim:
            st.markdown("#### 🧠 Análise do cenário")
            st.info(texto_exec)

    st.divider()
    st.subheader("📜 Recomendações de Cursos – CHP")

    # ==========================================================
    # CENÁRIO SIMULADO (DERIVADO DO SIMULADOR)
    # ==========================================================

    resultados_simulados = []

    for curso in df_matricula['CURSO'].unique():

        # ======================================================
        # CASO 1 — CURSO SELECIONADO (USA SIMULADOR)
        # ======================================================
        if curso == curso_sel:

            linha_simulada = linha_sim.copy(deep=True)
            linha_simulada['CURSO'] = curso
            linha_simulada['ANO'] = 2026

        # ======================================================
        # CASO 2 — OUTROS CURSOS (USA ÚLTIMO ANO)
        # ======================================================
        else:
            df_curso_global = df_matricula[
                df_matricula['CURSO'] == curso]

            if df_curso_global.empty:
                continue  # curso realmente inexistente

            # Base global do curso (CBO)
            linha_simulada = (
                df_curso_global
                .sort_values('ANO')
                .iloc[-1]
                .copy(deep=True)
                )
        
            linha_simulada['ANO'] = 2026
            linha_simulada['CURSO'] = curso
            linha_simulada['UNIDADE'] = unidade_sel

            # ✅ Variáveis dependentes da CBO → média das unidades
            linha_simulada['SALDO_EMPREGO'] = df_curso_global['SALDO_EMPREGO'].mean()
            linha_simulada['SALARIO_MEDIO'] = df_curso_global['SALARIO_MEDIO'].mean()

            # ✅ Variáveis locais → valores da unidade simulada
            linha_simulada['QTD_EMPRESAS'] = emp_sim
            linha_simulada['VLR_MEDIO_BENEFICIO'] = bf_sim
            linha_simulada['QTD_CONC'] = conc_sim

        # ======================================================
        # PREDIÇÃO
        # ======================================================
        X = build_X(linha_simulada)
        probs = modelo.predict_proba(X)[0]

        faixa_idx = int(np.argmax(probs))
        faixa_nome = FAIXAS[faixa_idx]
        prob_max = float(np.max(probs))

        recomendacao = formatar_recomendacao_celula(
            faixa_dominante=faixa_nome,
            prob_max=prob_max
        )

        resultados_simulados.append({
            'CURSO': curso,
            'RECOM_SIMULADO': recomendacao
        })

    df_simulado_2026 = pd.DataFrame(resultados_simulados)


    # ==========================================================
    # GERA MATRIZES FUTURAS POR CENÁRIO
    # ==========================================================

    df_base_2026 = gerar_matriz_curso_unidade_futuro(
        df_historico=df_matricula,
        ano_futuro=2026,
        cenario="base"
    )

    df_conservador_2026 = gerar_matriz_curso_unidade_futuro(
        df_historico=df_matricula,
        ano_futuro=2026,
        cenario="conservador"
    )

    df_otimista_2026 = gerar_matriz_curso_unidade_futuro(
        df_historico=df_matricula,
        ano_futuro=2026,
        cenario="otimista"
    )
   
    # ==========================================================
    # FILTRA PARA A UNIDADE SELECIONADA
    # ==========================================================
    #  Filtros (ANTES DE TUDO)
    unidades_sel = st.multiselect(
        "Unidades para Recomendações (CHP)",
        sorted(df_matricula['UNIDADE'].unique()),
        default=[unidade_sel]
    )

    cenarios_sel = st.multiselect(
        "Cenários para comparação",
        ['Base', 'Conservador', 'Otimista', 'Simulado'],
        default=['Base', 'Otimista', 'Simulado']
    )

    # 2️⃣ Filtra cenários por UNIDADE
    df_base_u = df_base_2026[df_base_2026['UNIDADE'].isin(unidades_sel)]
    df_cons_u = df_conservador_2026[df_conservador_2026['UNIDADE'].isin(unidades_sel)]
    df_oti_u  = df_otimista_2026[df_otimista_2026['UNIDADE'].isin(unidades_sel)]

    # 3️⃣ Consolida por CURSO
    df_base_u = df_base_u.groupby('CURSO', as_index=False).first()
    df_cons_u = df_cons_u.groupby('CURSO', as_index=False).first()
    df_oti_u  = df_oti_u.groupby('CURSO', as_index=False).first()

    # 4️⃣ Seleciona colunas e renomeia
    df_base_u = df_base_u[['CURSO', 'col_recom']].rename(columns={'col_recom': 'Base'})
    df_cons_u = df_cons_u[['CURSO', 'col_recom']].rename(columns={'col_recom': 'Conservador'})
    df_oti_u  = df_oti_u[['CURSO', 'col_recom']].rename(columns={'col_recom': 'Otimista'})

    # 5️⃣ Merge final
    df_recom_unidade = (
        df_base_u
        .merge(df_cons_u, on='CURSO', how='left')
        .merge(df_oti_u, on='CURSO', how='left')
        .merge(df_simulado_2026, on='CURSO', how='left')
    )

    # 6️⃣ Renomeia para exibição
    df_recom_unidade = df_recom_unidade.rename(columns={
        'CURSO': 'Curso Técnico',
        'Base': 'Base',
        'Conservador': 'Conservador',
        'Otimista': 'Otimista',
        'RECOM_SIMULADO': 'Simulado'
    })

    # 7️⃣ Aplica filtro de cenários
    colunas_finais = ['Curso Técnico']
    for c in cenarios_sel:
        if c in df_recom_unidade.columns:
            colunas_finais.append(c)

    df_recom_unidade = df_recom_unidade[colunas_finais]

    if 'UNIDADE' in df_recom_unidade.columns:
        df_recom_unidade = df_recom_unidade.drop(columns=['UNIDADE'])

    # ==========================================================
    # ORDENA: melhores recomendações primeiro
    # ==========================================================
    df_recom_unidade = df_recom_unidade.sort_values(by='Curso Técnico',ascending=True)

    # ==========================================================
    # EXIBE TABELA
    # ==========================================================

    st.markdown("""
        **Legenda das Recomendações:**

        - ✅ **Recomendar fortemente** – Alta probabilidade na faixa *Acima de 40*
        - 🟢 **Recomendar** – Probabilidade favorável na faixa *Acima de 40*
        - 🟡 **Oferta com cautela** – Faixa *Entre 21 e 40*
        - 🔴 **Não priorizar** – Faixa *Abaixo de 21*
        - ⚠️ **Analisar manualmente** – Cenário indefinido ou instável
        """)
    

    df_base_u = df_base_u.groupby('CURSO', as_index=False).first()
    df_cons_u = df_cons_u.groupby('CURSO', as_index=False).first()
    df_oti_u  = df_oti_u.groupby('CURSO', as_index=False).first()

    if 'UNIDADE' in df_recom_unidade.columns:
        df_recom_unidade = df_recom_unidade.drop(columns=['UNIDADE'])
    st.dataframe(df_recom_unidade, use_container_width=True, hide_index=True)

    # ---------- RODAPÉ ----------
    st.divider()
    st.caption(
        f"As probabilidades refletem o risco estimado considerando "
        f"simultaneamente **ANO**, **UNIDADE**, **CURSO** e **MUNICÍPIO**.")

# Inserir no filtro Curso e Município Aluno a opção de ter um global