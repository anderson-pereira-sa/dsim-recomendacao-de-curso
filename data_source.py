import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import unicodedata
import os
import glob
from sqlalchemy import create_engine

##################################################+ # + # + # + # + # + # + # + # + # + # + # + # + #
##### >>>>>>> Funções Gerais <<<<<<< #####
##########################################

# >>>>>>> - Função para consulta no banco de dados - <<<<<<< #
def consulta(query: str) -> pd.DataFrame:
    server = r"fbd101-001\HML"
    database = "DB_OBSERVATORIO"

    connection_string = (
        "mssql+pyodbc://@"
        f"{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    engine = create_engine(connection_string)

    try:
        return pd.read_sql_query(query, engine)
    except Exception as e:
        raise RuntimeError(f"Erro na consulta SQL: {e}")

# >>>>>>> - Função para limpar texto - <<<<<<< #
def limpar_texto(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.apply(lambda x: unicodedata.normalize('NFKD', x))
    s = s.apply(lambda x: "".join(ch for ch in x if not unicodedata.combining(ch)))
    s = s.str.replace(r"[^0-9A-Za-z]+", "", regex=True)
    s = s.str.upper()
    return s


##################################################+ # + # + # + # + # + # + # + # + # + # + # + # + #
##### >>>>>>> Carregamento dos dados <<<<<<< #####
##################################################

####################################################################################################
# >>>>>>> - MUNICIPIO - <<<<<<< #
municipio_unid = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\dsim-recomendacao-de-curso\GEO_UO_MAIS_PROXIMA.xlsx",
                                               sheet_name='BASE', header=0)

municipio_unid['MUNICIPIO_ORIGEM'] = municipio_unid['MUNICIPIO_ORIGEM'].str.upper()
municipio_unid['MUNICIPIO_UO_PRÓXIMA'] = municipio_unid['MUNICIPIO_UO_PRÓXIMA'].str.upper()

municipio_unid = municipio_unid.rename(columns={'MUNICIPIO_UO_PRÓXIMA':'UNIDADE',
                                                'MUNICIPIO_ORIGEM':'MUNICIPIO'},inplace=False)
municipio_unid['MUNICIPIO_COD'] = limpar_texto(municipio_unid['MUNICIPIO'])


####################################################################################################
# >>>>>>> - FONTE RAIS VINCULO - <<<<<<< #
df_rais_VINC = consulta('''
WITH rais_vinc AS
(
SELECT 
      COUNT(*) AS QTD_VINCULOS
      ,SUM(vinc_rais.VL_REMUNERACAO_MEDIA_NOMINAL) as SALARIO
      ,CASE WHEN dim_terri.NO_MUNICIPIO = 'CAMACARI' THEN 'CAMAÇARI'
              WHEN dim_terri.NO_MUNICIPIO = 'EUNAPOLIS' THEN 'EUNÁPOLIS'
              WHEN dim_terri.NO_MUNICIPIO = 'ILHEUS' THEN 'ILHÉUS'
              WHEN dim_terri.NO_MUNICIPIO = 'VITORIA DA CONQUISTA' THEN 'VITÓRIA DA CONQUISTA'
              WHEN dim_terri.NO_MUNICIPIO = 'JEQUIE' THEN 'JEQUIÉ'
              WHEN dim_terri.NO_MUNICIPIO = 'LUIS EDUARDO MAGALHAES' THEN 'LUÍS EDUARDO MAGALHÃES'
              WHEN dim_terri.NO_MUNICIPIO = 'SENHOR DO BONFIN' THEN 'SENHOR DO BONFIM'
              WHEN dim_terri.NO_MUNICIPIO = 'SANTO ANTONIO DE JESUS' THEN 'SANTO ANTÔNIO DE JESUS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO GONCALO DOS CAMPOS' THEN 'SÃO GONÇALO DOS CAMPOS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO SEBASTIAO DO PASSE' THEN 'SÃO SEBASTIÃO DO PASSE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FELIPE' THEN 'SÃO FELIPE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FRANCISCO DO CONDE' THEN 'SÃO FRANCISCO DO CONDE'
              WHEN dim_terri.NO_MUNICIPIO = 'SIMOES FILHO' THEN 'SIMÕES FILHO'
              ELSE dim_terri.NO_MUNICIPIO END AS NO_MUNICIPIO    
      ,vinc_rais.CO_CBO
      ,CASE WHEN vinc_rais.Ano = 2022 THEN 2023
            WHEN vinc_rais.Ano = 2023 THEN 2024
            WHEN vinc_rais.Ano = 2024 THEN 2025
      END AS ANO

FROM DB_OBSERVATORIO.rais.vinculos_bahia_historico vinc_rais
LEFT JOIN DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON vinc_rais.CO_MUN_IBGE_6 = dim_terri.CO_MUN_IBGE_6
WHERE Ano IN (2022, 2023, 2024)
GROUP BY dim_terri.NO_MUNICIPIO ,vinc_rais.Ano, vinc_rais.CO_CBO
)
SELECT 
      SUM(QTD_VINCULOS) AS QTD_VINCULOS
      ,SUM(SALARIO) / NULLIF(SUM(QTD_VINCULOS),0) AS SALARIO_MEDIO
      ,NO_MUNICIPIO
      ,CO_CBO
      ,ANO
FROM rais_vinc
GROUP BY NO_MUNICIPIO, CO_CBO, ANO
ORDER BY NO_MUNICIPIO DESC;
''')

df_rais_VINC['NO_MUNICIPIO_COD'] = limpar_texto(df_rais_VINC['NO_MUNICIPIO'])
df_rais_VINC_1 = df_rais_VINC.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_rais_VINC_1['CBO_UNIDADE'] = df_rais_VINC_1['CO_CBO'].astype(str) + df_rais_VINC_1['UNIDADE']
df_rais_VINC_2 = df_rais_VINC_1.groupby(['ANO','UNIDADE','CBO_UNIDADE'], as_index=False).agg({'QTD_VINCULOS':'sum','SALARIO_MEDIO':'mean'})
# df_rais_VINC_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_rais_vinculo.xlsx")


####################################################################################################
# >>>>>>> - FONTE RAIS ESTABELECIMENTO - <<<<<<< #
df_rais_ESTAB = consulta('''
WITH rais_estab AS
(
SELECT 
      COUNT(*) AS QTD_EMPRESAS
      ,CASE WHEN dim_terri.NO_MUNICIPIO = 'CAMACARI' THEN 'CAMAÇARI'
              WHEN dim_terri.NO_MUNICIPIO = 'EUNAPOLIS' THEN 'EUNÁPOLIS'
              WHEN dim_terri.NO_MUNICIPIO = 'ILHEUS' THEN 'ILHÉUS'
              WHEN dim_terri.NO_MUNICIPIO = 'VITORIA DA CONQUISTA' THEN 'VITÓRIA DA CONQUISTA'
              WHEN dim_terri.NO_MUNICIPIO = 'JEQUIE' THEN 'JEQUIÉ'
              WHEN dim_terri.NO_MUNICIPIO = 'LUIS EDUARDO MAGALHAES' THEN 'LUÍS EDUARDO MAGALHÃES'
              WHEN dim_terri.NO_MUNICIPIO = 'SENHOR DO BONFIN' THEN 'SENHOR DO BONFIM'
              WHEN dim_terri.NO_MUNICIPIO = 'SANTO ANTONIO DE JESUS' THEN 'SANTO ANTÔNIO DE JESUS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO GONCALO DOS CAMPOS' THEN 'SÃO GONÇALO DOS CAMPOS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO SEBASTIAO DO PASSE' THEN 'SÃO SEBASTIÃO DO PASSE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FELIPE' THEN 'SÃO FELIPE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FRANCISCO DO CONDE' THEN 'SÃO FRANCISCO DO CONDE'
              WHEN dim_terri.NO_MUNICIPIO = 'SIMOES FILHO' THEN 'SIMÕES FILHO'
              ELSE dim_terri.NO_MUNICIPIO END AS NO_MUNICIPIO    
      ,est_rais.COD_SUBCLASSE_CNAE
      ,CASE WHEN est_rais.Ano = 2022 THEN 2023
            WHEN est_rais.Ano = 2023 THEN 2024
            WHEN est_rais.Ano = 2024 THEN 2025
      END AS ANO
FROM DB_OBSERVATORIO.rais.estabelecimentos_bahia_historico est_rais
LEFT JOIN DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON est_rais.COD_MUN = dim_terri.CO_MUN_IBGE_6
WHERE Ano IN (2022, 2023, 2024)

GROUP BY -- est_rais.Qtd_Vinculos_Ativos
         dim_terri.NO_MUNICIPIO
         ,est_rais.COD_SUBCLASSE_CNAE
         ,est_rais.Ano
)
SELECT 
      SUM(QTD_EMPRESAS) AS QTD_EMPRESAS
      ,NO_MUNICIPIO
      ,ANO
FROM rais_estab
GROUP BY NO_MUNICIPIO, ANO
ORDER BY NO_MUNICIPIO DESC
;
''')

df_rais_ESTAB['NO_MUNICIPIO_COD'] = limpar_texto(df_rais_ESTAB['NO_MUNICIPIO'])
df_rais_ESTAB_1 = df_rais_ESTAB.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_rais_ESTAB_2 = df_rais_ESTAB_1.groupby(['ANO','UNIDADE'], as_index=False).agg({'QTD_EMPRESAS':'sum'})
# df_rais_ESTAB_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_rais_ESTAB_2.xlsx")


####################################################################################################
# >>>>>>> - FONTE CAGED - <<<<<<< #
df_caged = consulta('''
WITH CAGED_BAHIA_CTE AS
(
SELECT DISTINCT
      YEAR(Data) AS ANO
      ,municipio
      ,cbo2002_ocupacao as CO_CBO
      ,SUBSTRING(CAST(cbo2002_ocupacao AS VARCHAR(10)), 1, 4) AS CO_CBO_FAMILIA
      ,SUM(CASE WHEN desligados > 0 THEN salario ELSE 0 END) AS SALARIO_DESLIGADOS
      ,SUM(CASE WHEN admitidos > 0 THEN salario ELSE 0 END) AS SALARIO_ADMITIDOS
      ,AVG(salario) AS SALARIO_MEDIO
      ,SUM(admitidos) AS SUM_ADMITIDOS
      ,SUM(desligados) AS SUM_DESLIGADOS

FROM DB_OBSERVATORIO.caged.caged_bahia

WHERE YEAR(Data) IN (2023, 2024, 2025)
    AND salario BETWEEN 0.3*1518 AND 150 * 1518 -- Salário mínimo de 2025 é R$ 1.518,00
    AND categoria <> 111 -- Trabalhadores intermitentes
    -- AND indicador_aprendiz <> '1' -- Exclui aprendizes: SIM
    -- AND tam_estab_jan <> 98 -- Inválido
    -- AND tipo_movimentacao <> 60 -- Desligamento por morte 
GROUP BY YEAR(Data), municipio, cbo2002_ocupacao
)

SELECT caged_final.ANO, 
       CASE WHEN dim_terri.NO_MUNICIPIO = 'CAMACARI' THEN 'CAMAÇARI'
              WHEN dim_terri.NO_MUNICIPIO = 'EUNAPOLIS' THEN 'EUNÁPOLIS'
              WHEN dim_terri.NO_MUNICIPIO = 'ILHEUS' THEN 'ILHÉUS'
              WHEN dim_terri.NO_MUNICIPIO = 'VITORIA DA CONQUISTA' THEN 'VITÓRIA DA CONQUISTA'
              WHEN dim_terri.NO_MUNICIPIO = 'JEQUIE' THEN 'JEQUIÉ'
              WHEN dim_terri.NO_MUNICIPIO = 'LUIS EDUARDO MAGALHAES' THEN 'LUÍS EDUARDO MAGALHÃES'
              WHEN dim_terri.NO_MUNICIPIO = 'SENHOR DO BONFIN' THEN 'SENHOR DO BONFIM'
              WHEN dim_terri.NO_MUNICIPIO = 'SANTO ANTONIO DE JESUS' THEN 'SANTO ANTÔNIO DE JESUS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO GONCALO DOS CAMPOS' THEN 'SÃO GONÇALO DOS CAMPOS'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO SEBASTIAO DO PASSE' THEN 'SÃO SEBASTIÃO DO PASSE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FELIPE' THEN 'SÃO FELIPE'
              WHEN dim_terri.NO_MUNICIPIO = 'SAO FRANCISCO DO CONDE' THEN 'SÃO FRANCISCO DO CONDE'
              WHEN dim_terri.NO_MUNICIPIO = 'SIMOES FILHO' THEN 'SIMÕES FILHO'
              WHEN dim_terri.NO_MUNICIPIO = 'SANTA TEREZINHA' THEN 'SANTA TERESINHA'
              ELSE dim_terri.NO_MUNICIPIO END AS NO_MUNICIPIO, 
       caged_final.CO_CBO,
       caged_final.CO_CBO_FAMILIA,
       caged_final.SALARIO_MEDIO,
       caged_final.SALARIO_ADMITIDOS AS SALARIO_ADMITIDOS,
       caged_final.SALARIO_DESLIGADOS AS SALARIO_DESLIGADOS,
       caged_final.SUM_ADMITIDOS,
       caged_final.SUM_DESLIGADOS
         
FROM CAGED_BAHIA_CTE caged_final
LEFT JOIN DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON caged_final.municipio = dim_terri.CO_MUN_IBGE_6
;''')

df_caged['NO_MUNICIPIO_COD'] = limpar_texto(df_caged['NO_MUNICIPIO'])
df_caged_1 = df_caged.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_caged_1['CBO_UNIDADE'] = (df_caged_1['CO_CBO'] + df_caged_1['UNIDADE'])
df_caged_2 = df_caged_1.groupby(['ANO', 'UNIDADE', 'CO_CBO','CBO_UNIDADE'], as_index=False).agg({'SALARIO_ADMITIDOS':'mean', 
                                                                                                 'SALARIO_DESLIGADOS':'mean',
                                                                                                 'SALARIO_MEDIO':'mean',
                                                                                                 'SUM_ADMITIDOS':'sum',
                                                                                                 'SUM_DESLIGADOS':'sum'
                                                                                                 })
df_caged_2['SALDO_EMPREGO'] = df_caged_2['SUM_ADMITIDOS'] - df_caged_2['SUM_DESLIGADOS']
# df_caged_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_caged_2.xlsx")

####################################################################################################
# >>>>>>> - FONTE SISTEC - <<<<<<< #

caminho_pasta = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Base de Dados - PS\Cubo_CHP"

padrao_arquivos = os.path.join(caminho_pasta, "*.xlsx")
lista_arquivos = glob.glob(padrao_arquivos)

print(f"Arquivos encontrados: {len(lista_arquivos)}")
for arq in lista_arquivos:
    print(arq)


lista_df = []
for arquivo in lista_arquivos:
    df_temp = pd.read_excel(arquivo, sheet_name='Sheet', header=0)
    lista_df.append(df_temp)

dataset_CHP_PS = pd.concat(lista_df, ignore_index=True)

colunas_filtrar = [
    'TIPO_ALUNO', 'TIPO_MATRICULA', 'SITUACAO_CURSO', 'SITUAÇÃO_MAT_PERIODO_LETIVO', 'TIPO_GRATUIDADE','UNIDADE', 'CURSO',
    'MODALIDADE', 'TURNO', 'PERIODO_LETIVO', 'PERIODO_ALUNO', 'IDADE', 'CPF', 'RA', 'SEXO', 'CIDADE']

colunas_existentes = [c for c in colunas_filtrar if c in dataset_CHP_PS.columns]
mascara_valida = pd.Series([True] * len(dataset_CHP_PS), index=dataset_CHP_PS.index)

for col in colunas_existentes:
    serie_col = dataset_CHP_PS[col].astype(str).str.strip().str.lower()
    eh_total = serie_col.str.contains(r'\btotal\b', na = False)
    mascara_valida = mascara_valida & ~eh_total
dataset_CHP_PS = dataset_CHP_PS[mascara_valida].reset_index(drop=True)
cols_numericas = dataset_CHP_PS.select_dtypes(include=['number']).columns
dataset_CHP_PS[cols_numericas] = dataset_CHP_PS[cols_numericas].fillna(0)

condicoes = [
    dataset_CHP_PS["IDADE"].isna() | (dataset_CHP_PS["IDADE"] <= 14),
    dataset_CHP_PS["IDADE"] <= 17,
    dataset_CHP_PS["IDADE"] <= 24,
    dataset_CHP_PS["IDADE"] <= 30,
    dataset_CHP_PS["IDADE"] <= 39,
    dataset_CHP_PS["IDADE"] <= 50,
    dataset_CHP_PS["IDADE"] <= 59,
    dataset_CHP_PS["IDADE"] >= 60
]

faixas = [
    "Menor de 14 anos",
    "14 a 17",
    "18 a 24",
    "25 a 30",
    "31 a 39",
    "40 a 50",
    "51 a 59",
    "+ 60"
]

# dataset_CHP_PS["FAIXA_ETARIA"] = np.select(condicoes, faixas, default="Sem faixa")

dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.replace(r"^SENAI\s+", "", regex=True)
dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.replace(r"\bDENDEZEIROS\b|\bCIMATEC\b", "SALVADOR", regex=True)
dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.strip()
dataset_CHP_PS['ANO'] = dataset_CHP_PS['PERIODO_LETIVO'].astype(str).str[:4].astype(int)
dataset_CHP_PS['CPF_E_RA'] = dataset_CHP_PS['CPF'].astype(str) + dataset_CHP_PS['RA'].astype(str)
dataset_CHP_PS['MATRICULAS'] = 1
dataset_CHP_PS['CIDADE'] = dataset_CHP_PS['CIDADE'].str.strip().str.upper()
dataset_CHP_PS['CIDADE'] = np.where(dataset_CHP_PS['CIDADE'] == 'ARAÇAS', 'ARAÇÁS', 
                                        np.where(dataset_CHP_PS['CIDADE'] == 'SANTO ESTEVÃO', 'SANTO ESTÊVÃO', 
                                                 np.where(dataset_CHP_PS['CIDADE'] == 'SAO SEBASTIAO DO PASSE', 'SÃO SEBASTIÃO DO PASSÉ',
                                                          np.where(dataset_CHP_PS['CIDADE'] == 'NOVO HORIZONTE', 'CAMAÇARI',
                                                                   np.where(dataset_CHP_PS['CIDADE'] == 'PETROLINA', 'JUAZEIRO',
                                                                            np.where(dataset_CHP_PS['CIDADE'] == 'PORTO ALEGRE DO TOCANTINS', 'LUÍS EDUARDO MAGALHÃES',
                                                                                     np.where(dataset_CHP_PS['CIDADE'] == 'TAMBAÚ', 'CAMAÇARI',
                                                                                     dataset_CHP_PS['CIDADE'])))))))
dataset_CHP_PS['CIDADE_COD'] = limpar_texto(dataset_CHP_PS['CIDADE'])
dataset_CHP_PS = dataset_CHP_PS.drop(columns=['MUNICIPIO'], errors='ignore')

dataset_CHP_PS_0 = dataset_CHP_PS.merge(municipio_unid[['MUNICIPIO','MUNICIPIO_COD']], left_on='CIDADE_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','CIDADE', 'CIDADE_COD'], errors='ignore')

df_CHP = dataset_CHP_PS_0.groupby(['ANO', 'UNIDADE','CURSO'],as_index=False).agg({'MATRICULAS':'sum'})
# df_CHP = dataset_CHP_PS_0.groupby(['ANO', 'UNIDADE', 'MUNICIPIO','CURSO', 'TURNO','FAIXA_ETARIA'],as_index=False).agg({'MATRICULAS':'sum'})
df_CHP.columns = df_CHP.columns.str.strip()

ano_ref = 2025
df_CHP_Pagante_menor_ = df_CHP[df_CHP['ANO'] <= ano_ref]
df_CHP_Pagante_igual_ = df_CHP_Pagante_menor_.copy(deep=True)
df_CHP_Pagante_igual_['CURSO'] = df_CHP_Pagante_igual_['CURSO'].str.upper()
df_CHP_Pagante_igual_['CURSO'] = (df_CHP_Pagante_igual_['CURSO'].str.replace(r'^TÉCNICO\s+EM\s+', '', regex=True, flags=re.IGNORECASE).str.strip().str.upper())
df_CHP_Pagante_igual_['UNIDADE'] = df_CHP_Pagante_igual_['UNIDADE'].replace({'LEM':'LUÍS EDUARDO MAGALHÃES','FEIRA':'FEIRA DE SANTANA','CONQUISTA':'VITÓRIA DA CONQUISTA'})

####################################################################################################
# >>>>>>> - FONTE INEP - <<<<<<< #
df_curso_tec_inep = consulta('''
WITH INEP_BA_CTE AS (
SELECT
      inep_tec.NU_ANO_CENSO AS ANO
      ,inep_tec.CO_MUNICIPIO
      ,CASE WHEN inep_tec.TP_DEPENDENCIA = 4 THEN 'PRIVADA' ELSE 'PÚBLICA' END DEPEND_ADM
      ,inep_tec.NO_ENTIDADE
    --   ,inep_tec.CO_ENTIDADE
      ,UPPER(inep_tec.NO_CURSO_EDUC_PROFISSIONAL) CURSO
    --   ,inep_tec.QT_MAT_CURSO_TEC -- Matrículas em CTEP (Curso Técnico em Educação Profisisonal) Total
    -- Matrículas em CTEP = Educação Profissional + Concomitante + Subsequente:
      ,(inep_tec.QT_MAT_CURSO_TEC_CT + inep_tec.QT_MAT_CURSO_TEC_CONC + inep_tec.QT_MAT_TEC_SUBS) QTD_MAT
FROM DB_OBSERVATORIO.educacao.censo_escolar_ed_tecnica inep_tec
WHERE 
    inep_tec.TP_LOCALIZACAO_DIFERENCIADA in ('0') 
    AND inep_tec.NO_ENTIDADE NOT LIKE ('%SENAI%')
    AND inep_tec.NO_CURSO_EDUC_PROFISSIONAL IS NOT NULL
    AND UPPER(inep_tec.NO_CURSO_EDUC_PROFISSIONAL) NOT LIKE ('%MAGISTÉRIO%')
-- ORDER BY  inep_tec.NU_ANO_CENSO DESC
)
SELECT 
    inep.ANO,
    CASE WHEN dim_terri.NO_MUNICIPIO = 'CAMACARI' THEN 'CAMAÇARI'
        WHEN dim_terri.NO_MUNICIPIO = 'EUNAPOLIS' THEN 'EUNÁPOLIS'
        WHEN dim_terri.NO_MUNICIPIO = 'ILHEUS' THEN 'ILHÉUS'
        WHEN dim_terri.NO_MUNICIPIO = 'VITORIA DA CONQUISTA' THEN 'VITÓRIA DA CONQUISTA'
        WHEN dim_terri.NO_MUNICIPIO = 'JEQUIE' THEN 'JEQUIÉ'
        WHEN dim_terri.NO_MUNICIPIO = 'LUIS EDUARDO MAGALHAES' THEN 'LUÍS EDUARDO MAGALHÃES'
        WHEN dim_terri.NO_MUNICIPIO = 'SENHOR DO BONFIN' THEN 'SENHOR DO BONFIM'
        WHEN dim_terri.NO_MUNICIPIO = 'SANTO ANTONIO DE JESUS' THEN 'SANTO ANTÔNIO DE JESUS'
        WHEN dim_terri.NO_MUNICIPIO = 'SAO GONCALO DOS CAMPOS' THEN 'SÃO GONÇALO DOS CAMPOS'
        WHEN dim_terri.NO_MUNICIPIO = 'SAO SEBASTIAO DO PASSE' THEN 'SÃO SEBASTIÃO DO PASSE'
        WHEN dim_terri.NO_MUNICIPIO = 'SAO FELIPE' THEN 'SÃO FELIPE'
        WHEN dim_terri.NO_MUNICIPIO = 'SAO FRANCISCO DO CONDE' THEN 'SÃO FRANCISCO DO CONDE'
        WHEN dim_terri.NO_MUNICIPIO = 'SIMOES FILHO' THEN 'SIMÕES FILHO' ELSE dim_terri.NO_MUNICIPIO END AS NO_MUNICIPIO,
    inep.CURSO,
    CASE
        WHEN inep.NO_ENTIDADE = 'CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL' THEN 'CEEP'
        WHEN inep.NO_ENTIDADE = 'CENTRO EST DE ED PROFISSIONAL' THEN 'CEEP'
        WHEN inep.NO_ENTIDADE = 'CENTRO EST DE EDUC PROFISSIONAL' THEN 'CEEP'
        WHEN inep.NO_ENTIDADE = 'CENTRO TERRITORIAL DE EDUCAÇÃO PROFISSIONAL' THEN 'CTEP'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCAÇÃO PROFISSIONAL COMÉRCIO' THEN 'SENAC'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCAÇÃO PROFISSIONAL SENAC CASA DO COMÉRCIO' THEN 'SENAC'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCAÇÃO PROFISSIONAL SENAC LAURO DE FREITAS' THEN 'SENAC'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCAÇÃO PROFISSIONAL DE ALAGOINHAS CEP ALH' THEN 'SENAC'
        WHEN inep.NO_ENTIDADE LIKE '%SENAC%' THEN 'SENAC'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCACAO PROFISSIONAL' THEN 'CEP'
        WHEN inep.NO_ENTIDADE = 'CENTRO EDUCACIONAL MUNICIPAL' THEN 'CEM'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EDUCACAO' THEN 'CE'
        WHEN inep.NO_ENTIDADE = 'CENTRO EDUCACIONAL' THEN 'CE'
        WHEN inep.NO_ENTIDADE = 'CASA FAMILIAR AGROFLORESTAL' THEN 'CFA'
        WHEN inep.NO_ENTIDADE = 'CASA FAMILIAR RURAL' THEN 'CFR'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE ENSINO GRAU T UNIDADE CAJAZEIRAS' THEN 'GRAU TÉCNICO'
        WHEN inep.NO_ENTIDADE LIKE '%GRAU TÉCNICO%' THEN 'GRAU TÉCNICO'
        WHEN inep.NO_ENTIDADE LIKE '%GRAUTECNICO%' THEN 'GRAU TÉCNICO'
        WHEN inep.NO_ENTIDADE LIKE '%GRAU TECNICO%' THEN 'GRAU TÉCNICO'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE EXCELÊNCIA EM FRUTICULTURA' THEN 'SENAR'
        WHEN inep.NO_ENTIDADE LIKE '%SENART%' THEN 'SENART'
        WHEN inep.NO_ENTIDADE LIKE '%CRUZEIRO DO SUL%' THEN 'UNICSUL'
        WHEN inep.NO_ENTIDADE LIKE '%UNICSUL%' THEN 'UNICSUL'
        WHEN inep.NO_ENTIDADE LIKE '%SE7E CENTRO%' THEN 'SE7E CENTRO TECNOLÓGICO'
        WHEN inep.NO_ENTIDADE = 'PROCURSOS SERVIÇOS DE EDUCAÇÃO' THEN 'PROCURSOS'
        WHEN inep.NO_ENTIDADE = 'CENTRO DE ESTUDOS TÉCNICOS, TREINAMENTOS PROFISSIONAIS E SERVIÇOS - CETTPS' THEN 'CETTP''S - CETTPS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL CARLITO DE CARVALHO%' THEN 'COLEGIO ESTADUAL CARLITO DE CARVALHO'
        WHEN inep.NO_ENTIDADE LIKE '%CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL AGUAS%' THEN 'CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL AGUAS'
        WHEN inep.NO_ENTIDADE LIKE '%CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL DO SEMI ARIDO%' THEN 'CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL DO SEMI ARIDO'
        WHEN inep.NO_ENTIDADE LIKE '%CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DA BACIA DO JACUIPE IIIEDNA DALTRO EM TEMPO INTEGRAL%' THEN 'CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DA BACIA DO JACUIPE IIIEDNA DALTRO'
        WHEN inep.NO_ENTIDADE LIKE '%CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DA CHAPADA DIAMANTINA TEMPO INTEGRAL%' THEN 'CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DA CHAPADA DIAMANTINA'
        WHEN inep.NO_ENTIDADE LIKE '%CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DE MEDEIROS NETO TEMPO INTEGRAL%' THEN 'CENTRO TERRITORIAL DE EDUCACAO PROFISSIONAL DE MEDEIROS NETO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO DEMOCRATICO ESTADUAL ANISIO TEIXEIRA TEMPO INTEGRAL%' THEN 'COLEGIO DEMOCRATICO ESTADUAL ANISIO TEIXEIRA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL ANTONIO CARLOS MAGALHAES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL ANTONIO CARLOS MAGALHAES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL ANTONIO RODRIGUES VIANA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL ANTONIO RODRIGUES VIANA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL ARISTIDES CEDRAZ DE OLIVEIRA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL ARISTIDES CEDRAZ DE OLIVEIRA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL CONCEICAO DO JACUIPE TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL CONCEICAO DO JACUIPE'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE BOQUIRA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE BOQUIRA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE COCOS TEMPO INTEGRAL' THEN 'COLEGIO ESTADUAL DE COCOS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE CONCEICAO DA FEIRA CECF TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE CONCEICAO DA FEIRA CECF'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE CORRENTINA %' THEN 'COLEGIO ESTADUAL DE CORRENTINA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE IBIQUERA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE IBIQUERA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE JEQUIE TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE JEQUIE'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE MALHADA DE PEDRAS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE MALHADA DE PEDRAS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DE TEMPO INTEGRAL RUI BARBOSA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DE RUI BARBOSA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DEMOCRATICO QUITERIA MARIA DE JESUS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DEMOCRATICO QUITERIA MARIA DE JESUS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DINAH GONCALVES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DINAH GONCALVES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO CAMPO ANNA JUNQUEIRA AYRES TOURINHO TEMPO INTEGRAL DISTRITO DE MATARIPE%' THEN 'COLEGIO ESTADUAL DO CAMPO ANNA JUNQUEIRA AYRES TOURINHO DISTRITO DE MATARIPE'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO CAMPO DE CARNAIBA TEMPO INTEGRAL DISTRITO DE CARNAIBA%' THEN 'COLEGIO ESTADUAL DO CAMPO DE CARNAIBA DISTRITO DE CARNAIBA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO CAMPO MIGUEL MOREIRA DE CARVALHO DIST RODA VELHA%' THEN 'COLEGIO ESTADUAL DO CAMPO MIGUEL MOREIRA DE CARVALHO DIST RODA VELHA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO CAMPO JORGE CALMON DISTRITO DE OLIVENCA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DO CAMPO JORGE CALMON DISTRITO DE OLIVENCA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO CAMPO MIGUEL MOREIRA DE CARVALHO DIST RODA VELHA%' THEN 'COLEGIO ESTADUAL DO CAMPO MIGUEL MOREIRA DE CARVALHO DISTRITO DE RODA VELHA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DO STIEP CARLOS MARIGHELLA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DO STIEP CARLOS MARIGHELLA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DOM PEDRO I TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DOM PEDRO I'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DR ANTONIO CARLOS MAGALHAES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DR ANTONIO CARLOS MAGALHAES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DOUTOR IVES ORLANDO LOPES DA SILVA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DOUTOR IVES ORLANDO LOPES DA SILVA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL DOUTOR JOSE ANTONIO DE ARAUJO PIMENTA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL DOUTOR JOSE ANTONIO DE ARAUJO PIMENTA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL EDVALDO FLORES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL EDVALDO FLORES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL ERALDO TINOCO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL ERALDO TINOCO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL ERNESTO CARNEIRO RIBEIRO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL ERNESTO CARNEIRO RIBEIRO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL EVANDRO BRANDAO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL EVANDRO BRANDAO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL GENTIL PARAISO MARTINS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL GENTIL PARAISO MARTINS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL GOVERNADOR LUIZ VIANA FILHO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL GOVERNADOR LUIZ VIANA FILHO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL GRANDES MESTRES BRASILEIROS EM TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL GRANDES MESTRES BRASILEIROS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL GRANDES MESTRES BRASILEIROS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL GRANDES MESTRES BRASILEIROS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOAO BENEVIDES NOGUEIRA EM TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOAO BENEVIDES NOGUEIRA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOAO BENEVIDES NOGUEIRA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOAO BENEVIDES NOGUEIRA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JORGE CALMON DISTRITO DE OLIVENCA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JORGE CALMON DISTRITO DE OLIVENCA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE DANTAS DE SOUZA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE DANTAS DE SOUZA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE DE FREITAS MASCARENHAS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE DE FREITAS MASCARENHAS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE RIBEIRO DE ARAUJO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE RIBEIRO DE ARAUJO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE RIBEIRO PAMPONET TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE RIBEIRO PAMPONET'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE VICENTE LEAL EM TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE VICENTE LEAL'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL JOSE VICENTE LEAL TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL JOSE VICENTE LEAL'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL LUIS EDUARDO MAGALHAES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL LUIS EDUARDO MAGALHAES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL LUIS NAVARRO DE BRITO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL LUIS NAVARRO DE BRITO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL LUIZ ROGERIO DE SOUZA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL LUIZ ROGERIO DE SOUZA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL LUIZ VIANA FILHO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL LUIZ VIANA FILHO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL LUIZ VIANA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL LUIZ VIANA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MANDINHO DE SOUZA ALMEIDA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MANDINHO DE SOUZA ALMEIDA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MANOEL FRANCISCO DE CAIRES EM TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MANOEL FRANCISCO DE CAIRES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MARIA OTILIA LUTZ - BAIRRO JARDIM DAS ACACIAS - TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MARIA OTILIA LUTZ - BAIRRO JARDIM DAS ACACIAS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MARIA OTILIA LUTZ BAIRRO JARDIM DAS ACACIAS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MARIA OTILIA LUTZ BAIRRO JARDIM DAS ACACIAS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MINISTRO OLIVEIRA BRITO - TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MINISTRO OLIVEIRA BRITO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MINISTRO OLIVEIRA BRITO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MINISTRO OLIVEIRA BRITO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL MONSENHOR TURIBIO VILANOVA TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL MONSENHOR TURIBIO VILANOVA'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL NEMISIA RIBEIRO DOS SANTOS TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL NEMISIA RIBEIRO DOS SANTOS'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL NERCY ANTONIO DUARTE TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL NERCY ANTONIO DUARTE'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL NOSSA SENHORA DA CONCEICAO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL NOSSA SENHORA DA CONCEICAO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL NOSSA SENHORA DO ROSARIO TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL NOSSA SENHORA DO ROSARIO'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL OCTACILIO MANOEL GOMES TEMPO INTEGRAL%' THEN 'COLEGIO ESTADUAL OCTACILIO MANOEL GOMES'
        WHEN inep.NO_ENTIDADE LIKE '%COLEGIO ESTADUAL OLAVO ALVES PINTO%' THEN 'COLEGIO ESTADUAL OLAVO ALVES PINTO'

        ELSE inep.NO_ENTIDADE END AS NO_ENTIDADE,
    
    SUM(inep.QTD_MAT) QTD_MAT_CONC

FROM 
    INEP_BA_CTE inep

LEFT JOIN 
    DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON inep.CO_MUNICIPIO = dim_terri.CO_MUN_IBGE_7
GROUP BY 
   inep.ANO, dim_terri.NO_MUNICIPIO, inep.CURSO, inep.NO_ENTIDADE
HAVING 
   SUM(inep.QTD_MAT) > 0
ORDER BY  inep.ANO DESC
''')

df_curso_tec_inep['NO_MUNICIPIO_COD'] = limpar_texto(df_curso_tec_inep['NO_MUNICIPIO'])
df_curso_tec_inep_1 = df_curso_tec_inep.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_curso_tec_inep_2 = df_curso_tec_inep_1.groupby(['ANO', 'CURSO','UNIDADE','NO_ENTIDADE'], as_index=False).agg({'NO_ENTIDADE':'nunique', 
                                                                                            'QTD_MAT_CONC':'mean'})
df_curso_tec_inep_2.rename(columns={'NO_ENTIDADE':'QTD_CONC'}, inplace=True)

df_curso_tec_inep_2['QTD_CONC'] = df_curso_tec_inep_2['QTD_CONC'].astype(int)
df_curso_tec_inep_2['QTD_MAT_CONC'] = df_curso_tec_inep_2['QTD_MAT_CONC'].astype(int)
# df_curso_tec_inep_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_curso_tec_inep_2.xlsx")


####################################################################################################
# >>>>>>> - FONTE SISTEC - <<<<<<< #
files = r'C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\Dados_Concorrencia_TS'
excel_files = glob.glob(os.path.join(files, '*.xlsx'))

dataframes = []
for file in excel_files:
    df = pd.read_excel(file)
    dataframes.append(df)
concorrencia = pd.concat(dataframes, ignore_index=True)
concorrencia = concorrencia[~concorrencia['UNIDADE_DE_ENSINO'].str.contains('SENAI',case=False, na=False)]
concorrencia['MODALIDADE'] = concorrencia['MODALIDADE'].replace({'Educação Presencial':'PRES','Educação a Distância':'EAD'})
concorrencia['ANO'] = pd.to_datetime(concorrencia['Dt_SCRAPING'], dayfirst=True).dt.year
concorrencia['ANO'] = concorrencia['ANO'].replace({2026: 2025})
concorrencia.drop(columns=['Dt_SCRAPING','AZURE_GEOLOCATOR'], inplace=True)
concorrencia['QTD_CONC'] = 1
concorrencia = concorrencia.drop_duplicates()
concorrencia_1 = concorrencia.merge(municipio_unid[['MUNICIPIO','UNIDADE']], on= 'MUNICIPIO', how='left')

concorrencia_1['UNIDADE_DE_ENSINO_AJUSTADO'] = (
        concorrencia_1['UNIDADE_DE_ENSINO']
        .str.upper()
        .str.replace(r'^\d+', '', regex=True)
        .str.replace(r'[-–/]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

substituicoes = {
        'CENTRO ESTADUAL DE EDUCACAO PROFISSIONAL': 'CEEP',
        'CENTRO EST DE ED PROFISSIONAL':'CEEP',
        'CENTRO EST DE EDUC PROFISSIONAL': 'CEEP',
        'CENTRO TERRITORIAL DE EDUCAÇÃO PROFISSIONAL':'CTEP',
        'CENTRO DE EDUCAÇÃO PROFISSIONAL COMÉRCIO':'SENAC',
        'CENTRO DE EDUCAÇÃO PROFISSIONAL SENAC CASA DO COMÉRCIO': 'SENAC',
        'CENTRO DE EDUCAÇÃO PROFISSIONAL SENAC LAURO DE FREITAS' : 'SENAC',
        'CENTRO DE EDUCAÇÃO PROFISSIONAL DE ALAGOINHAS CEP ALH': 'SENAC',
        'SENAC CA':'SENAC',
        'SENAC CAMAÇARI':'SENAC',
        'SENAC CEP FS': 'SENAC',    
        'SENAC CEP PITUBA': 'SENAC',
        'SENAC CEP PS': 'SENAC',
        'SENAC CEP SANTO ANTONIO DE JESUS': 'SENAC',
        'SENAC CEP VC': 'SENAC',
        'SENAC FEIRA DE SANTANA': 'SENAC',
        'SENAC SÉ': 'SENAC',
        'CENTRO DE EDUCACAO PROFISSIONAL': 'CEP',
        'CENTRO EDUCACIONAL MUNICIPAL': 'CEM',
        'CENTRO DE EDUCACAO': 'CE',
        'CENTRO EDUCACIONAL': 'CE',
        'CASA FAMILIAR AGROFLORESTAL': 'CFA',
        'CASA FAMILIAR RURAL': 'CFR',
        'CENTRO DE ENSINO GRAU T UNIDADE CAJAZEIRAS': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO FEIRA DE SANTANA BA': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNID VITÓRIA DA CONQUISTA': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNIDADE CAMAÇARI': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNIDADE PAULO AFONSO BA': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNIDADE PLATAFORMA': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNIDADE SÃO CRISTÓVÃO': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO JUAZEIRO': 'GRAU TÉCNICO',
        'CENTRO DE ENSINO GRAU TÉCNICO UNIDADE FONTE NOVA': 'GRAU TÉCNICO',
        'GRAU TÉCNICO JUAZEIRO': 'GRAU TÉCNICO',
        'GRAU TÉCNICO UNIDADE FONTE NOVA': 'GRAU TÉCNICO',
        'CENTRO DE EXCELÊNCIA EM FRUTICULTURA' : 'SENAR',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL BROTAS_MATATU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL CAMAÇARI (ALTO DA CRUZ)': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL CONCEIÇÃO DO JACUÍPE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL IPIAÚ _CENTRO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL ITACARÉ': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL JEREMOABO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL MORRO DO CHAPÉU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL PIRITIBA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POJUCA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ALAGOINHAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ARACI': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO BARRA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO BARRA DA ESTIVA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO BARREIRAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CACHOEIRA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CAMACAN': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CAMAMU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CAMAÇARI': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CAMAÇARI_AREMBEPE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CANAVIEIRAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CANDIDO SALES': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CATU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CONCEICAO DO COITE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO CONCEIÇAO DA FEIRA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ENTRE RIOS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO EUCLIDES DA CUNHA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO EUNAPOLIS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO FEIRA DE SANTANA_CID. NOVA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO GANDU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO GUARATINGA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO IBOTIRAMA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ILHEUS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO INHAMBUPE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO IRECE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ITAPETINGA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO ITUBERA_NOBERTO ODEBRECHT': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO JACOBINA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO JOAO DOURADO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO JUAZEIRO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO LUIS EDUARDO MAGALHAES': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO MARACAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO MUCURI_ITABATA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO NOVA VICOSA_POSTO DA MATA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO OLINDINA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO OLIVEIRA DOS BREJINHOS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO PILAO ARCADO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO PLANALTO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO PORTO SEGURO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO PRADO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO PRESIDENTE TANCREDO NEVES': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO RIBEIRA DO POMBAL': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO RIO REAL': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_CENTENARIO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_LIBERDADE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_LOBATO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_NAZARE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_PARIPE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_ROMA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SALVADOR_SAO CRISTOVAO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SANTA MARIA DA VITÓRIA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SANTALUZ': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SANTO ANTONIO DE JESUS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SANTO ESTEVAO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SATIRO DIAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SERRINHA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO SIMOES FILHO': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO TANHACU': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO TEIXEIRA DE FREITAS': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO TEOFILANDIA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO VALENCA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO VALENTE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL POLO VITORIA DA CONQUISTA': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL PORTO SEGURO (PARQUE ECOLÓGICO)': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL VERA CRUZ': 'UNICSUL',
        'UNICSUL_AREMBEPE': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL CAMAÇARI (ALTO DA CRUZ)': 'UNICSUL',
        'UNIVERSIDADE CRUZEIRO DO SUL UNICSUL PORTO SEGURO (PARQUE ECOLÓGICO)': 'UNICSUL',
        'SE7E CENTRO TECNOLÓGICO REMANSO': 'SE7E CENTRO TECNOLÓGICO',
        'SE7E CENTRO TECNOLÓGICO SALVADOR': 'SE7E CENTRO TECNOLÓGICO',
        'PROCURSOS SERVIÇOS DE EDUCAÇÃO':'PROCURSOS',
        'CENTRO DE ESTUDOS TÉCNICOS, TREINAMENTOS PROFISSIONAIS E SERVIÇOS - CETTPS':"CETTP'S - CETTPS",
    }
concorrencia_1['UNIDADE_DE_ENSINO_AJUSTADO'] = concorrencia_1['UNIDADE_DE_ENSINO_AJUSTADO'].replace(substituicoes, regex=True)
concorrencia_1.sort_values(by=['UNIDADE_DE_ENSINO_AJUSTADO','CURSO','MUNICIPIO','MODALIDADE'], inplace=True)
concorrencia_sistec = concorrencia_1.groupby(['ANO', 'CURSO','UNIDADE'], as_index=False).agg({'UNIDADE_DE_ENSINO_AJUSTADO':'nunique'})
concorrencia_sistec.rename(columns={'UNIDADE_DE_ENSINO_AJUSTADO':'QTD_CONC'}, inplace=True)
concorrencia_sistec['QTD_MAT_CONC'] = 0


####################################################################################################
# >>>>>>> - FONTE BOLSA FAMILIA - <<<<<<< #
df_bolsaf = consulta('''
SELECT
      year(bolsa.referencia) as ANO
      ,bolsa.familias_pbf_pos_2023 as QTD_FAMILIAS
      ,bolsa.valor_repasse_familias_pbf_a_partir_2023 as TOTAL_REPASSE
      ,bolsa.valor_beneficio_medio_apos_mar_2023 as VLR_MEDIO_BENEFICIO
      ,CASE WHEN dim_terri.NO_MUNICIPIO = 'CAMACARI' THEN 'CAMAÇARI'
            WHEN dim_terri.NO_MUNICIPIO = 'EUNAPOLIS' THEN 'EUNÁPOLIS'
            WHEN dim_terri.NO_MUNICIPIO = 'ILHEUS' THEN 'ILHÉUS'
            WHEN dim_terri.NO_MUNICIPIO = 'VITORIA DA CONQUISTA' THEN 'VITÓRIA DA CONQUISTA'
            WHEN dim_terri.NO_MUNICIPIO = 'JEQUIE' THEN 'JEQUIÉ'
            WHEN dim_terri.NO_MUNICIPIO = 'LUIS EDUARDO MAGALHAES' THEN 'LUÍS EDUARDO MAGALHÃES'
            WHEN dim_terri.NO_MUNICIPIO = 'SENHOR DO BONFIN' THEN 'SENHOR DO BONFIM'
            WHEN dim_terri.NO_MUNICIPIO = 'SANTO ANTONIO DE JESUS' THEN 'SANTO ANTÔNIO DE JESUS'
            WHEN dim_terri.NO_MUNICIPIO = 'SAO GONCALO DOS CAMPOS' THEN 'SÃO GONÇALO DOS CAMPOS'
            WHEN dim_terri.NO_MUNICIPIO = 'SAO SEBASTIAO DO PASSE' THEN 'SÃO SEBASTIÃO DO PASSE'
            WHEN dim_terri.NO_MUNICIPIO = 'SAO FELIPE' THEN 'SÃO FELIPE'
            WHEN dim_terri.NO_MUNICIPIO = 'SAO FRANCISCO DO CONDE' THEN 'SÃO FRANCISCO DO CONDE'
            WHEN dim_terri.NO_MUNICIPIO = 'SIMOES FILHO' THEN 'SIMÕES FILHO' ELSE dim_terri.NO_MUNICIPIO END AS NO_MUNICIPIO  
FROM DB_OBSERVATORIO.cadunico.bolsa_familia_familias bolsa
LEFT JOIN DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON bolsa.CO_MUN_IBGE_6 = dim_terri.CO_MUN_IBGE_6
WHERE YEAR(bolsa.referencia) IN (2023, 2024, 2025) AND bolsa.uf = 'BA'
ORDER BY dim_terri.NO_MUNICIPIO, year(bolsa.referencia)
;
''')

df_bolsaf['NO_MUNICIPIO_COD'] = limpar_texto(df_bolsaf['NO_MUNICIPIO'])
df_bolsaf_1 = df_bolsaf.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_bolsaf_2 = df_bolsaf_1.groupby(['ANO','UNIDADE'], as_index=False).agg({'QTD_FAMILIAS':'sum', 
                                                                          'TOTAL_REPASSE':'sum', 
                                                                          'VLR_MEDIO_BENEFICIO':'mean'})
# df_bolsaf_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\dsim-recomendacao-de-curso\df_bolsaf_2.xlsx")


####################################################################################################
# >>>>>>> - FONTE CURSOS TÉCNICOS SENAI [CUBO] - <<<<<<< #
caminho_pasta = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Base de Dados - PS\Cubo_CHP"

padrao_arquivos = os.path.join(caminho_pasta, "*.xlsx")
lista_arquivos = glob.glob(padrao_arquivos)

print(f"Arquivos encontrados: {len(lista_arquivos)}")
for arq in lista_arquivos:
    print(arq)

lista_df = []
for arquivo in lista_arquivos:
    df_temp = pd.read_excel(arquivo, sheet_name='Sheet', header=0)
    lista_df.append(df_temp)

dataset_CHP_PS = pd.concat(lista_df, ignore_index=True)
colunas_filtrar = [
    'TIPO_ALUNO', 'TIPO_MATRICULA', 'SITUACAO_CURSO', 'SITUAÇÃO_MAT_PERIODO_LETIVO', 'TIPO_GRATUIDADE','UNIDADE', 'CURSO',
    'MODALIDADE', 'TURNO', 'PERIODO_LETIVO', 'PERIODO_ALUNO', 'IDADE', 'CPF', 'RA', 'SEXO', 'CIDADE']

colunas_existentes = [c for c in colunas_filtrar if c in dataset_CHP_PS.columns]
mascara_valida = pd.Series([True] * len(dataset_CHP_PS), index=dataset_CHP_PS.index)

for col in colunas_existentes:
    serie_col = dataset_CHP_PS[col].astype(str).str.strip().str.lower()
    eh_total = serie_col.str.contains(r'\btotal\b', na = False)
    mascara_valida = mascara_valida & ~eh_total
dataset_CHP_PS = dataset_CHP_PS[mascara_valida].reset_index(drop=True)
cols_numericas = dataset_CHP_PS.select_dtypes(include=['number']).columns
dataset_CHP_PS[cols_numericas] = dataset_CHP_PS[cols_numericas].fillna(0)

condicoes = [
    dataset_CHP_PS["IDADE"].isna() | (dataset_CHP_PS["IDADE"] <= 14),
    dataset_CHP_PS["IDADE"] <= 17,
    dataset_CHP_PS["IDADE"] <= 24,
    dataset_CHP_PS["IDADE"] <= 30,
    dataset_CHP_PS["IDADE"] <= 39,
    dataset_CHP_PS["IDADE"] <= 50,
    dataset_CHP_PS["IDADE"] <= 59,
    dataset_CHP_PS["IDADE"] >= 60
]

faixas = [
    "Menor de 14 anos",
    "14 a 17",
    "18 a 24",
    "25 a 30",
    "31 a 39",
    "40 a 50",
    "51 a 59",
    "+ 60"
]

dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.replace(r"^SENAI\s+", "", regex=True)
dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.replace(r"\bDENDEZEIROS\b|\bCIMATEC\b", "SALVADOR", regex=True)
dataset_CHP_PS["UNIDADE"] = dataset_CHP_PS["UNIDADE"].str.strip()
dataset_CHP_PS['ANO'] = dataset_CHP_PS['PERIODO_LETIVO'].astype(str).str[:4].astype(int)
dataset_CHP_PS['CPF_E_RA'] = dataset_CHP_PS['CPF'].astype(str) + dataset_CHP_PS['RA'].astype(str)
dataset_CHP_PS['MATRICULAS'] = 1
dataset_CHP_PS['CIDADE'] = dataset_CHP_PS['CIDADE'].str.strip().str.upper()
dataset_CHP_PS['CIDADE'] = np.where(dataset_CHP_PS['CIDADE'] == 'ARAÇAS', 'ARAÇÁS', 
                                        np.where(dataset_CHP_PS['CIDADE'] == 'SANTO ESTEVÃO', 'SANTO ESTÊVÃO', 
                                                 np.where(dataset_CHP_PS['CIDADE'] == 'SAO SEBASTIAO DO PASSE', 'SÃO SEBASTIÃO DO PASSÉ',
                                                          np.where(dataset_CHP_PS['CIDADE'] == 'NOVO HORIZONTE', 'CAMAÇARI',
                                                                   np.where(dataset_CHP_PS['CIDADE'] == 'PETROLINA', 'JUAZEIRO',
                                                                            np.where(dataset_CHP_PS['CIDADE'] == 'PORTO ALEGRE DO TOCANTINS', 'LUÍS EDUARDO MAGALHÃES',
                                                                                     np.where(dataset_CHP_PS['CIDADE'] == 'TAMBAÚ', 'CAMAÇARI',
                                                                                     dataset_CHP_PS['CIDADE'])))))))
dataset_CHP_PS['CIDADE_COD'] = limpar_texto(dataset_CHP_PS['CIDADE'])
dataset_CHP_PS = dataset_CHP_PS.drop(columns=['MUNICIPIO'], errors='ignore')

dataset_CHP_PS_0 = dataset_CHP_PS.merge(municipio_unid[['MUNICIPIO','MUNICIPIO_COD']], left_on='CIDADE_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','CIDADE', 'CIDADE_COD'], errors='ignore')

df_CHP = dataset_CHP_PS_0.groupby(['ANO', 'UNIDADE','CURSO'],as_index=False).agg({'MATRICULAS':'sum'})
# df_CHP = dataset_CHP_PS_0.groupby(['ANO', 'UNIDADE', 'MUNICIPIO','CURSO', 'TURNO','FAIXA_ETARIA'],as_index=False).agg({'MATRICULAS':'sum'})
df_CHP.columns = df_CHP.columns.str.strip()

ano_ref = 2025
df_CHP_Pagante_menor_ = df_CHP[df_CHP['ANO'] <= ano_ref]
df_CHP_Pagante_igual_ = df_CHP_Pagante_menor_.copy(deep=True)
df_CHP_Pagante_igual_['CURSO'] = df_CHP_Pagante_igual_['CURSO'].str.upper()
df_CHP_Pagante_igual_['CURSO'] = (df_CHP_Pagante_igual_['CURSO'].str.replace(r'^TÉCNICO\s+EM\s+', '', regex=True, flags=re.IGNORECASE).str.strip().str.upper())
df_CHP_Pagante_igual_['UNIDADE'] = df_CHP_Pagante_igual_['UNIDADE'].replace({'LEM':'LUÍS EDUARDO MAGALHÃES','FEIRA':'FEIRA DE SANTANA','CONQUISTA':'VITÓRIA DA CONQUISTA'})


####################################################################################################
# >>>>>>> - FONTE CURSOS TÉCNICOS SENAI [ITINERARIO - CBO] - <<<<<<< #
flle_iti = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\dsim-recomendacao-de-curso\dados_extraidos_final.xlsx"
itinerario = pd.read_excel(flle_iti, sheet_name='Sheet1')
itinerario['MODALIDADE'] = np.where(itinerario['Nome do Item'].str.contains('Aperfeiçoamento|Especialista|Especialização', case=False, na=False), 'CAEP', 'CHP')
itinerario['Nome do Item'] = itinerario['Nome do Item'].str.upper()
itinerario['Nome do Item'] = itinerario['Nome do Item'].replace({'TÉCNICO EM ':'',
                                                             'TECNÓLOGO EM ':''}, regex=True)
itinerario_1 = itinerario[itinerario['MODALIDADE'] == 'CHP'].drop_duplicates().reset_index(drop=True)



####################################################################################################
# >>>>>>> - FONTE CATÁLOGO NACIONAL DE CURSOS TÉCNICOS (CNCT) - <<<<<<< #
flle_cnct = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\dsim-recomendacao-de-curso\catalogo_cnct.xlsx"
df_catalogo_cnct  = pd.read_excel(flle_cnct, sheet_name='catalogo_cnct')
df_catalogo_cnct.drop(columns=['Eixo Tecnológico','Área Tecnológica','Perfil Profissional de Conclusão','Carga Horária Mínima',
                               'Pré-Requisitos para Ingresso','Descrição Carga Horária Mínima','Campo de Atuação'], inplace=True, errors='ignore')
df_catalogo_cnct['Denominação do Curso'] = df_catalogo_cnct['Denominação do Curso'].str.upper()
df_catalogo_cnct['Denominação do Curso'] = df_catalogo_cnct['Denominação do Curso'].replace({'TÉCNICO EM ':'','TECNÓLOGO EM ':''}, regex=True)

df_catalogo_cnct_cbo = df_catalogo_cnct[['Denominação do Curso', 'Ocupações CBO Associadas']].drop_duplicates().dropna().reset_index(drop=True)
df_catalogo_cnct_itinerario = df_catalogo_cnct[['Denominação do Curso', 'Itinerários Formativos']].drop_duplicates().dropna().reset_index(drop=True)

colunas_multilinha_cbo = ['Ocupações CBO Associadas']
colunas_multilinha_itinerario = ['Itinerários Formativos']

def explode_multiplas_colunas(linha, colunas):
    listas = {col: str(linha[col]).split('\n') if pd.notna(linha[col]) else [''] for col in colunas}
    max_len = max(len(v) for v in listas.values())
    for col, lista in listas.items():
        if len(lista) < max_len:
            listas[col] = lista + [''] * (max_len - len(lista))
    df_expandido = pd.DataFrame(listas)
    for col in linha.index:
        if col not in colunas:
            df_expandido[col] = linha[col]  
    return df_expandido

df_catalogo_cnct_cbo = pd.concat(df_catalogo_cnct_cbo.apply(explode_multiplas_colunas,
                                                    axis=1, colunas=colunas_multilinha_cbo).to_list(),
                                                    ignore_index=True)
df_catalogo_cnct_cbo = df_catalogo_cnct_cbo[df_catalogo_cnct_cbo['Ocupações CBO Associadas'].astype(str).str.strip().ne('') & 
                                            df_catalogo_cnct_cbo['Ocupações CBO Associadas'].notna() & 
                                            ~df_catalogo_cnct_cbo['Ocupações CBO Associadas'].str.contains('Ocupação ainda não classificada', case=False, na=False)].reset_index(drop=True)

df_catalogo_cnct_cbo['CBO'] = (
    df_catalogo_cnct_cbo['Ocupações CBO Associadas']
        .astype(str)
        .str.extract(r'(\d{4}\s*-\s*\d{2}|\d{6}|\d{4})')[0]
        .str.replace(r'\D', '', regex=True)
)

df_catalogo_cnct_itinerario = pd.concat(df_catalogo_cnct_itinerario.apply(explode_multiplas_colunas,
                                                    axis=1, colunas=colunas_multilinha_itinerario).to_list(),
                                                    ignore_index=True)

df_catalogo_cnct_itinerario['Itinerários Formativos'] = np.where(df_catalogo_cnct_itinerario['Itinerários Formativos'].str.contains('Curso Superior de Tecnologia em Agroecologia', case=False, na=False),'Curso Superior de Tecnologia em Agroecologia',
                                                                 np.where(df_catalogo_cnct_itinerario['Itinerários Formativos'].str.contains('Bacharealdo em Negócios da Moda', case=False, na=False), 'Bacharelado em Negócios da Moda',
                                                                          df_catalogo_cnct_itinerario['Itinerários Formativos']))

df_catalogo_cnct_itinerario['MODALIDADE'] = np.where(df_catalogo_cnct_itinerario['Itinerários Formativos'].str.contains('Especialização Técnica em ', case=False, na=False), 'POSTEC',
                                                 np.where(df_catalogo_cnct_itinerario['Itinerários Formativos'
                                                                                      ].str.contains('Bacharelado |Bacharel |Licenciatura |Curso Superior de Tecnologia |Graduação |Bacharelado/Licenciatura ', 
                                                                                                                             case=False, na=False), 'GRAD', 'FIC'))

df_catalogo_cnct_itinerario = df_catalogo_cnct_itinerario[~df_catalogo_cnct_itinerario['Itinerários Formativos'].str.contains('Sugestões de |Não identificadas|Não há|O curso não prevê|:', case=False, na=False)].reset_index(drop=True)
df_catalogo_cnct_itinerario = df_catalogo_cnct_itinerario[df_catalogo_cnct_itinerario['Itinerários Formativos'].astype(str).str.strip().ne('') 
                                                          & df_catalogo_cnct_itinerario['Itinerários Formativos'].notna()].reset_index(drop=True)

for col in colunas_multilinha_cbo:
    df_catalogo_cnct_cbo[col] = df_catalogo_cnct_cbo[col].str.strip()

for col in colunas_multilinha_itinerario:
    df_catalogo_cnct_itinerario[col] = df_catalogo_cnct_itinerario[col].str.strip()

df_catalogo_cnct_1 = df_catalogo_cnct[['Denominação do Curso']].merge(df_catalogo_cnct_cbo, on='Denominação do Curso', how='left'
                                                                      ).merge(df_catalogo_cnct_itinerario, on='Denominação do Curso', how='left')

df_catalogo_cnct_1_tec = df_catalogo_cnct_1[['CBO', 'Denominação do Curso']].copy(deep=True)
df_catalogo_cnct_1_tec.loc[:, 'MODALIDADE'] = 'CHP'
df_catalogo_cnct_1_tec.rename(columns={'Denominação do Curso':'CURSO'}, inplace=True)

df_catalogo_cnct_1_outros = df_catalogo_cnct_1[['CBO','Itinerários Formativos','MODALIDADE']].copy(deep=True)
df_catalogo_cnct_1_outros.rename(columns={'Itinerários Formativos':'CURSO'}, inplace=True)
df_catalogo_cnct_1_outros = df_catalogo_cnct_1_outros[df_catalogo_cnct_1_outros['CURSO'].notna()].reset_index(drop=True)

df_catalogo_cnct_2 = pd.concat([df_catalogo_cnct_1_tec, df_catalogo_cnct_1_outros],axis=0,ignore_index=True)
df_catalogo_cnct_2 = df_catalogo_cnct_2[df_catalogo_cnct_2['CBO'].notna()].reset_index(drop=True)
df_catalogo_cnct_2.drop_duplicates(inplace=True)
df_catalogo_cnct_2=df_catalogo_cnct_2[df_catalogo_cnct_2['MODALIDADE'] == 'CHP'].reset_index(drop=True)


##################################################+ # + # + # + # + # + # + # + # + # + # + # + # + #
##### >>>>>>> ETL <<<<<<< #####
##################################################

# >>>>>>> - CHP PAGANTE - <<<<<<< #
df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_.copy(deep=True)
df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_2.groupby(['ANO','CURSO','UNIDADE'], as_index=False).agg({'MATRICULAS':'sum'})

# >>>>>>> - RAIS ESTABELECIMENTO - <<<<<<< #
df_CHP_Pagante_igual_2= df_CHP_Pagante_igual_2.merge(df_rais_ESTAB_2[['ANO','UNIDADE','QTD_EMPRESAS']],
                                                           how='left',
                                                           left_on=['UNIDADE','ANO'],
                                                           right_on=['UNIDADE','ANO'])

df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_2.rename(columns={'MATRICULAS':'MAT_PAG'})

# >>>>>>> - ITINERÁRIO / CNCT - <<<<<<< #
join_CHP_itinerario_2025_1 = df_CHP_Pagante_igual_2.merge(df_catalogo_cnct_2[['CURSO','CBO']], 
                                                          how='left',
                                                          left_on='CURSO', 
                                                          right_on='CURSO')

# join_CHP_itinerario_2025_1.drop(columns='Nome do Item', inplace=True)
join_CHP_itinerario_2025_1['CBO_familia'] = join_CHP_itinerario_2025_1['CBO'].str.split('-').str[0]
join_CHP_itinerario_2025_1['CBO'] = join_CHP_itinerario_2025_1['CBO'].str.replace('-','',regex=False)
join_CHP_itinerario_2025_1['CBO_UNIDADE'] = join_CHP_itinerario_2025_1['CBO'] + join_CHP_itinerario_2025_1['UNIDADE']

# >>>>>>> - CAGED - <<<<<<< #
join_CHP_itinerario_2025_2_caged = join_CHP_itinerario_2025_1.merge(
    df_caged_2[['ANO','CBO_UNIDADE', 'SALARIO_MEDIO', 'SUM_ADMITIDOS','SUM_DESLIGADOS','SALDO_EMPREGO']],
                    how='left', 
                    left_on=['CBO_UNIDADE','ANO'], 
                    right_on=['CBO_UNIDADE','ANO'])

# >>>>>>> - RAIS VINCULOS - <<<<<<< #
join_CHP_itinerario_2025_3_caged= join_CHP_itinerario_2025_2_caged.merge(df_rais_VINC_2[['ANO','CBO_UNIDADE','QTD_VINCULOS']], # SALARIO_MEDIO
                                                           how='left',
                                                           left_on=['CBO_UNIDADE','ANO'],
                                                           right_on=['CBO_UNIDADE','ANO'])

# >>>>>>> - CONCORRÊNCIA INEP + SISTEC - <<<<<<< #
df_curso_tec_inep_3 = pd.concat([df_curso_tec_inep_2, concorrencia_sistec], axis=0, ignore_index=True)

join_CHP_itinerario_2025_4_caged = join_CHP_itinerario_2025_3_caged.merge(df_curso_tec_inep_3[['ANO','UNIDADE','CURSO','QTD_CONC','QTD_MAT_CONC']],
                                                           how='left',
                                                           left_on=['UNIDADE','ANO','CURSO'],
                                                           right_on=['UNIDADE','ANO','CURSO'])

join_CHP_itinerario_2025_5_caged = join_CHP_itinerario_2025_4_caged.merge(df_bolsaf_2[['ANO','UNIDADE','QTD_FAMILIAS','TOTAL_REPASSE','VLR_MEDIO_BENEFICIO']],
                                                           how='left',
                                                           left_on=['UNIDADE','ANO'],
                                                           right_on=['UNIDADE','ANO'])

# >>>>>>> - BOLSA FAMILIA - <<<<<<< #
join_CHP_itinerario_2025_5_caged = join_CHP_itinerario_2025_4_caged.merge(df_bolsaf_2[['ANO','UNIDADE','QTD_FAMILIAS','TOTAL_REPASSE','VLR_MEDIO_BENEFICIO']],
                                                           how='left',
                                                           left_on=['UNIDADE','ANO'],
                                                           right_on=['UNIDADE','ANO'])

# >>>>>>> - DATAFRAME FINAL - <<<<<<< #
join_CHP_itinerario_2025_6_caged = join_CHP_itinerario_2025_5_caged.groupby(['ANO','CURSO', 'UNIDADE', 
                                                                     ], as_index=False).agg({'MAT_PAG': 'sum',
                                                                                             'QTD_CONC': 'sum', 
                                                                                             'QTD_MAT_CONC': 'sum', 
                                                                                             'QTD_EMPRESAS': 'sum', 
                                                                                             'QTD_VINCULOS': 'sum', 
                                                                                             'SUM_ADMITIDOS': 'sum', 
                                                                                             'SUM_DESLIGADOS': 'sum', 
                                                                                             'SALDO_EMPREGO': 'sum', 
                                                                                             'SALARIO_MEDIO': 'mean',
                                                                                             'QTD_FAMILIAS':'sum',
                                                                                             'TOTAL_REPASSE':'sum',
                                                                                             'VLR_MEDIO_BENEFICIO':'mean'})

join_CHP_itinerario_2025_7_caged = join_CHP_itinerario_2025_6_caged.copy(deep=True)
join_CHP_itinerario_2025_7_caged = join_CHP_itinerario_2025_7_caged[(join_CHP_itinerario_2025_7_caged['SALDO_EMPREGO'].notnull()) & (join_CHP_itinerario_2025_7_caged['QTD_VINCULOS'].notnull())].reset_index(drop=True)

labels_faixas = ['Abaixo de 21', 'Entre 21 e 40', 'Acima de 40']
join_CHP_itinerario_2025_7_caged['FAIXA_MAT'] = pd.cut(join_CHP_itinerario_2025_7_caged['MAT_PAG'],
                                                       bins=[-np.inf, 20, 40, np.inf],
                                                       labels=labels_faixas,
                                                       right=True)

cols_num = join_CHP_itinerario_2025_7_caged.select_dtypes(include='number').columns
join_CHP_itinerario_2025_7_caged[cols_num] = (join_CHP_itinerario_2025_7_caged[cols_num].fillna(0))

join_CHP_itinerario_2025_7_caged.to_pickle(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\dsim-recomendacao-de-curso\dataset.pkl")
# print(join_CHP_itinerario_2025_7_caged.head(5))
print("Dataset final processado e salvo como dataset.pkl")
