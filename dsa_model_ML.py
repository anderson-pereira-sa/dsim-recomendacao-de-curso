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
    layout="wide"
)

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


def impacto_variaveis_locais(linha_real, linha_sim, faixa_idx, delta_padrao=0.10, epsilon=1.0):
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


def texto_executivo_dinamico(faixa_nome, impactos_dict):

    impactos_ord = pd.Series(impactos_dict).sort_values(key=abs, ascending=False)

    var_principal = impactos_ord.index[0]
    impacto_principal = impactos_ord.iloc[0]
    var_label = labels_exibicao.get(var_principal, var_principal)

    mag = abs(impacto_principal) * 100

    if mag >= 20:
        tipo = "mudança estrutural"
    elif mag >= 5:
        tipo = "impacto relevante"
    else:
        tipo = "ajuste marginal"

    if impacto_principal < 0:
        direcao = "não reforça"
        efeito = "desloca fortemente o cenário para outras faixas"
    else:
        direcao = "reforça"
        efeito = "aumenta a probabilidade dessa faixa"

    return (
        f"Nesse cenário específico, a variável **{var_label}** é o principal fator explicativo. "
        f"O seu aumento **{direcao}** a faixa **{faixa_nome}** e "
        f"{efeito}, caracterizando uma **{tipo}** no perfil de demanda."
    )

def probabilidades_por_ano(df_base, unidade, curso):
    resultados = []

    df_base_contexto = (
        df_base[(df_base['UNIDADE'] == unidade) &
                ((curso == 'GLOBAL') | (df_base['CURSO'] == curso))].sort_values('ANO'))

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

def grafico_real_barras_horizontais(probs_atual, ano_atual):
    df_bar = pd.DataFrame({
        "Faixa": FAIXAS,
        "Probabilidade": probs_atual
    })

    cores = {
        FAIXAS[0]: "#1D3691",  # Abaixo ou igual a 20
        FAIXAS[1]: "#795821",  # Entre 21 e 40
        FAIXAS[2]: "#195E32"   # Acima ou igual a 41
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
                size=18        # ✅ AQUI você controla o tamanho
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
        FAIXAS[0]: "#1D3691",  # Abaixo ou igual a 20
        FAIXAS[1]: "#795821",  # Entre 21 e 40
        FAIXAS[2]: "#195E32"   # Acima ou igual a 41
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
                    shape="spline"  # ✅ suavização visual
                ),
                marker=dict(size=6)
            )
        )

    fig.update_layout(
        title="Evolução Histórica das Probabilidades por Faixa",
        xaxis_title="Ano",
        yaxis_title="Probabilidade",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=360,
        legend_title_text="Faixa de Matrícula"
    )

    return fig

def analise_executiva_prob_real(
    faixa_dominante,
    prob_dominante,
    impactos_dict,
    limite_otimo=0.80,
    limite_relevante_pp=1.0
):

    # Ordena impactos por magnitude
    impactos_ord = (
        pd.Series(impactos_dict)
        .sort_values(key=abs, ascending=False)
    )

    # Variáveis com impacto negativo relevante
    impactos_negativos = impactos_ord[
        impactos_ord < -limite_relevante_pp / 100
    ]

    # =========================
    # CENÁRIO RUIM (≤20)
    # =========================
    if faixa_dominante == "Abaixo ou igual a 20":
        texto = (
            "O cenário atual apresenta **alto risco de baixa demanda**, com maior "
            "probabilidade concentrada na faixa **abaixo ou igual a 20 matrículas**. "
            "Esse resultado indica necessidade de **intervenção estratégica**. "
            "Recomenda-se utilizar o **simulador** para avaliar melhorias em variáveis "
            "estruturais e de mercado, buscando deslocar o cenário para faixas superiores."
        )
        tipo = "warning"
        return texto, tipo

    # =========================
    # CENÁRIO INTERMEDIÁRIO (21–40)
    # =========================
    if faixa_dominante == "Entre 21 e 40":
        texto = (
            "O cenário atual indica uma **situação intermediária de demanda**, com maior "
            "probabilidade concentrada na faixa **entre 21 e 40 matrículas**. "
            "Há **potencial de crescimento**, e ajustes táticos no mercado local, "
            "na atratividade do curso ou na oferta institucional podem contribuir "
            "para evolução para a faixa superior."
        )
        tipo = "info"
        return texto, tipo

    # =========================
    # CENÁRIO ÓTIMO (≥41)
    # =========================
    if faixa_dominante == "Acima ou igual a 41" and prob_dominante >= limite_otimo:

        if not impactos_negativos.empty:
            # Lista das principais variáveis de risco
            vars_risco = [
                f"{labels_exibicao.get(v, v)}"
                for v in impactos_negativos.index[:2]
            ]

            texto = (
                "Estamos em um **cenário excelente de demanda**, com probabilidade "
                "elevada de permanência na faixa **acima ou igual a 41 matrículas**. "
                "No entanto, o modelo indica que **movimentos adicionais no mercado de trabalho**, "
                f"especialmente relacionados a **{', '.join(vars_risco)}**, "
                "podem **deslocar parte da demanda para o emprego direto**, "
                "reduzindo marginalmente a probabilidade de matrícula muito elevada. "
                "A recomendação é **manter a estratégia atual** e **monitorar o mercado**."
            )
        else:
            texto = (
                "O cenário atual é **altamente favorável**, com elevada probabilidade "
                "de permanência na faixa **acima ou igual a 41 matrículas**. "
                "Não há sinais relevantes de risco no curto prazo. "
                "A recomendação é **manter a estratégia vigente** e acompanhar a evolução "
                "dos indicadores de mercado."
            )

        tipo = "success"
        return texto, tipo

    # =========================
    # FALLBACK (segurança)
    # =========================
    texto = (
        "O cenário atual apresenta distribuição equilibrada das probabilidades. "
        "Recomenda-se análise complementar por meio do simulador."
    )
    tipo = "info"
    return texto, tipo


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

# ==========================================================
# ABAS
# ==========================================================

tab1, tab2 = st.tabs(["📌 Contexto do Modelo", "🔎 Painel de Decisão"])

# ==========================================================
# ABA 1
# ==========================================================

with tab1:

    st.subheader("📌 Sobre o Modelo")

    st.markdown("""
    ⁂ Este modelo utiliza técnicas de machine learning para 
    apoiar decisões estratégicas relacionadas à recomendação de curso por matrículas:

       **📍Abaixo ou igual a 20 matrículas |📍 Entre 21 e 40 matrículas |📍 Acima ou igual a 41 matrículas**

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

    st.sidebar.header("🎯 Contexto da Análise")

    # ANO
    ano_max = int(df_matricula['ANO'].max())
    anos_disponiveis = sorted(df_matricula['ANO'].unique())

    ano_sel = st.sidebar.selectbox(
        "ANO",
        anos_disponiveis,
        index=anos_disponiveis.index(ano_max)
    )

    df_ano = df_matricula[df_matricula['ANO'] == ano_sel]

    # UNIDADE
    unidade_sel = st.sidebar.selectbox(
        "UNIDADE",
        sorted(df_ano['UNIDADE'].unique())
    )

    df_unidade = df_ano[df_ano['UNIDADE'] == unidade_sel]

    # CURSO (COM GLOBAL)
    lista_cursos = sorted(df_unidade['CURSO'].unique())
    curso_sel = st.sidebar.selectbox(
        "CURSO",
        ['GLOBAL'] + lista_cursos)

    if curso_sel == 'GLOBAL':
        df_curso = df_unidade.copy(deep=True)
    else:
        df_curso = df_unidade[df_unidade['CURSO'] == curso_sel]

    # ==========================================================
    # LINHA REPRESENTATIVA (MESMA BASE DA TABELA)
    # ==========================================================

    if curso_sel == 'GLOBAL':
        linha_real = (df_matricula[(df_matricula['UNIDADE'] == unidade_sel)].sort_values(['ANO','CURSO']).iloc[-1].copy(deep=True))
    else:
        linha_real = (df_matricula[
                (df_matricula['UNIDADE'] == unidade_sel) &
                (df_matricula['CURSO'] == curso_sel)].sort_values(['ANO','CURSO']).iloc[-1].copy(deep=True))

    # ---------- PROBABILIDADE REAL ----------
    probs_real = modelo.predict_proba(build_X(linha_real))[0]

    # ==========================================================
    # LAYOUT: REAL | DIVISOR | SIMULADO
    # ==========================================================

    col_real, col_div, col_sim = st.columns([1.8, 0.05, 1.2])

    # ---------- REAL ----------
    with col_real:

        st.subheader("📊 Cenário REAL")

        g_bar, texto = st.columns(2)

        with g_bar:
            st.plotly_chart(
                grafico_real_barras_horizontais(
                    probs_atual=probs_real,
                    ano_atual=ano_sel
                ),
                use_container_width=True
            )

        with texto:
            st.markdown("#### ✦ Probabilidade REAL")
            faixa_dominante = FAIXAS[int(np.argmax(probs_real))]
            prob_dominante = float(np.max(probs_real))
            
            impactos = impacto_variaveis_locais(linha_real, linha_real, int(np.argmax(probs_real)))
            texto_prob_real, tipo_prob_real = analise_executiva_prob_real(
                faixa_dominante=faixa_dominante,
                prob_dominante=prob_dominante,
                impactos_dict=impactos  # ← saída de impacto_variaveis_locais
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

        st.plotly_chart(
            grafico_real_linhas(df_series),
            use_container_width=True
        )

        # ---------- TABELA HISTÓRICA ----------
        st.subheader("📋 Contexto do Município Selecionado")

        df_contexto_historico = (
            df_matricula[
                (df_matricula['UNIDADE'] == unidade_sel) &
                ((curso_sel == 'GLOBAL') | (df_matricula['CURSO'] == curso_sel))
            ]
            .sort_values('ANO')
        )
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

        # with c2:
            # matc_sim = st.number_input(labels_exibicao['QTD_MAT_CONC'], value=int(linha_real['QTD_MAT_CONC']))
            # vinc_sim = st.number_input(labels_exibicao['QTD_VINCULOS'], value=int(linha_real['QTD_VINCULOS']))

        with c2:
            sal_sim = st.number_input(labels_exibicao['SALARIO_MEDIO'], value=float(linha_real['SALARIO_MEDIO']))
            saldo_sim = st.number_input(labels_exibicao['SALDO_EMPREGO'], value=int(linha_real['SALDO_EMPREGO']))

        linha_sim = linha_real.copy(deep=True)
        valores_simulados = {
            'QTD_EMPRESAS': emp_sim,
            # 'QTD_VINCULOS': vinc_sim,
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
        texto_exec = texto_executivo_dinamico(faixa_nome, impactos)

        st.plotly_chart(pizza_sim(probs_sim, "Distribuição SIMULADA"), True)

        var_sim, an_sim = st.columns(2)

        with var_sim:
            st.markdown("#### ✦ Variáveis mais influentes")
            for v, i in pd.Series(impactos).sort_values(key=abs, ascending=False).head(5).items():
                st.write(f"▶ **{labels_exibicao.get(v, v)}** ({i*100:+.1f} p.p.)")

        with an_sim:
            st.markdown("#### 🧠 Análise do cenário")
            st.info(texto_exec)

    # ---------- RODAPÉ ----------
    st.divider()
    st.caption(
        f"As probabilidades refletem o risco estimado considerando "
        f"simultaneamente **ano {ano_sel}**, UNIDADE, CURSO e MUNICÍPIO."
    )


# Inserir no filtro Curso e Município Aluno a opção de ter um global