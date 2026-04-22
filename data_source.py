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
        "?driver=ODBC Driver 17 for SQL Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    engine = create_engine(connection_string)

    try:
        df = pd.read_sql_query(query, engine)
        return df
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
municipio_unid = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\ds-im-modelo-de-recomendacao-de-curso\GEO_UO_MAIS_PROXIMA.xlsx",
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
      -- ,AVG(vinc_rais.VL_REMUNERACAO_DEZEMBRO_NOMINAL) as SALARIO
      ,AVG(vinc_rais.VL_REMUNERACAO_MEDIA_NOMINAL) as SALARIO
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
WHERE 
 --Ano = (SELECT MAX(Ano) FROM DB_OBSERVATORIO.rais.vinculos_bahia_historico)
   Ano IN (2022, 2023, 2024)
GROUP BY -- vinc_rais.Qtd_Vinculos_Ativos
         dim_terri.NO_MUNICIPIO
        -- ,vinc_rais.VL_REMUNERACAO_DEZEMBRO_NOMINAL
         ,vinc_rais.Ano, vinc_rais.CO_CBO
)
SELECT 
      SUM(QTD_VINCULOS) AS QTD_VINCULOS
      ,AVG(SALARIO) AS SALARIO_MEDIO
      ,NO_MUNICIPIO
      ,CO_CBO
      ,ANO
FROM rais_vinc
GROUP BY NO_MUNICIPIO, CO_CBO, ANO
ORDER BY NO_MUNICIPIO DESC
;
''')

df_rais_VINC['NO_MUNICIPIO_COD'] = limpar_texto(df_rais_VINC['NO_MUNICIPIO'])
df_rais_VINC_1 = df_rais_VINC.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_rais_VINC_1['CBO_MUNICIPIO'] = df_rais_VINC_1['CO_CBO'].astype(str) + df_rais_VINC_1['MUNICIPIO']
df_rais_VINC_2 = df_rais_VINC_1.groupby(['ANO','MUNICIPIO','CBO_MUNICIPIO'], as_index=False).agg({'QTD_VINCULOS':'sum','SALARIO_MEDIO':'mean'})
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
WHERE 
  -- Ano = (SELECT MAX(Ano) FROM DB_OBSERVATORIO.rais.estabelecimentos_bahia_historico)
  Ano IN (2022, 2023, 2024)

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
df_rais_ESTAB_2 = df_rais_ESTAB_1.groupby(['ANO','MUNICIPIO'], as_index=False).agg({'QTD_EMPRESAS':'sum'})
# df_rais_ESTAB_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_rais_ESTAB_2.xlsx")


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
    --   ,UPPER(inep_tec.NO_CURSO_EDUC_PROFISSIONAL) CURSO_INEP
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
    -- inep.CURSO_INEP,
    COUNT(inep.NO_ENTIDADE) QTD_CONC_INEP,
    SUM(inep.QTD_MAT) QTD_MAT_CONC

FROM 
    INEP_BA_CTE inep
LEFT JOIN 
    DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON inep.CO_MUNICIPIO = dim_terri.CO_MUN_IBGE_7
GROUP BY 
    inep.ANO, dim_terri.NO_MUNICIPIO -- , inep.CURSO_INEP
HAVING 
    SUM(inep.QTD_MAT) > 0
-- ORDER BY  inep.ANO DESC
''')
df_curso_tec_inep['NO_MUNICIPIO_COD'] = limpar_texto(df_curso_tec_inep['NO_MUNICIPIO'])
df_curso_tec_inep_1 = df_curso_tec_inep.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_curso_tec_inep_2 = df_curso_tec_inep_1.groupby(['ANO', 'MUNICIPIO'], as_index=False).agg({'QTD_CONC_INEP':'mean', 
                                                                                            'QTD_MAT_CONC':'mean'})
# df_curso_tec_inep_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_curso_tec_inep_2.xlsx")


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
      ,CASE WHEN desligados > 0 THEN salario ELSE 0 END AS SALARIO_DESLIGADOS
      ,CASE WHEN admitidos > 0 THEN salario ELSE 0 END AS SALARIO_ADMITIDOS
      ,admitidos
      ,desligados
FROM DB_OBSERVATORIO.caged.caged_bahia

WHERE YEAR(Data) IN (2023, 2024, 2025)
    AND valor_salario_fixo BETWEEN 0.3*1518 AND 150 * 1518 -- Salário mínimo de 2025 é R$ 1.518,00
    AND categoria <> 111 -- Trabalhadores intermitentes
    -- AND indicador_aprendiz <> '1' -- Exclui aprendizes: SIM
    -- AND tam_estab_jan <> 98 -- Inválido
    -- AND tipo_movimentacao <> 60 -- Desligamento por morte 
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
       AVG(caged_final.SALARIO_ADMITIDOS) AS SALARIO_ADMITIDOS,
       AVG(caged_final.SALARIO_DESLIGADOS) AS SALARIO_DESLIGADOS,
       SUM(caged_final.admitidos) AS SUM_ADMITIDOS,
       SUM(caged_final.desligados) AS SUM_DESLIGADOS,
       SUM( caged_final.admitidos - caged_final.desligados) AS SALDO_EMPREGO
    --    COALESCE( AVG(caged_final.SALARIO_ADMITIDOS) / NULLIF(AVG(caged_final.SALARIO_DESLIGADOS), 0), 1) AS PRESSAO_SALARIAL
         
FROM CAGED_BAHIA_CTE caged_final
LEFT JOIN DB_OBSERVATORIO.referencia.dimensao_territorio_territorio_identidade dim_terri ON caged_final.municipio = dim_terri.CO_MUN_IBGE_6
GROUP BY caged_final.ANO,  dim_terri.NO_MUNICIPIO, caged_final.CO_CBO, caged_final.CO_CBO_FAMILIA
HAVING 
    COALESCE( AVG(caged_final.SALARIO_ADMITIDOS) / NULLIF(AVG(caged_final.SALARIO_DESLIGADOS), 0), 1) > 0

ORDER BY NO_MUNICIPIO, CO_CBO DESC
;''')

df_caged['NO_MUNICIPIO_COD'] = limpar_texto(df_caged['NO_MUNICIPIO'])
df_caged_1 = df_caged.merge(municipio_unid[['MUNICIPIO','UNIDADE','MUNICIPIO_COD']], 
                            left_on='NO_MUNICIPIO_COD', right_on='MUNICIPIO_COD', how='left').drop(
                                columns=['MUNICIPIO_COD','NO_MUNICIPIO', 'NO_MUNICIPIO_COD'], errors='ignore')
df_caged_1['CBO_MUNICIPIO'] = (df_caged_1['CO_CBO'] + df_caged_1['MUNICIPIO'])
df_caged_2 = df_caged_1.groupby(['ANO', 'MUNICIPIO', 'CO_CBO','CBO_MUNICIPIO'], as_index=False).agg({'SALARIO_ADMITIDOS':'mean', 
                                                                                                 'SALARIO_DESLIGADOS':'mean',
                                                                                                 'SUM_ADMITIDOS':'sum',
                                                                                                 'SUM_DESLIGADOS':'sum',
                                                                                                 'SALDO_EMPREGO':'sum'})
df_caged_2['PRESSAO_SALARIAL'] = df_caged_2.apply(lambda row: row['SALARIO_ADMITIDOS'] / row['SALARIO_DESLIGADOS'] if row['SALARIO_DESLIGADOS'] > 0 else 0, axis=1)
# df_caged_2 = pd.read_excel(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\df_caged_2.xlsx")


####################################################################################################
# >>>>>>> - FONTE CURSOS TÉCNICOS SENAI [CUBO] - <<<<<<< #
caminho_pasta = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\Cubo_CHP"

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
    dataset_CHP_PS["IDADE"] >= 60]

faixas = [
    "Menor de 14 anos",
    "14 a 17",
    "18 a 24",
    "25 a 30",
    "31 a 39",
    "40 a 50",
    "51 a 59",
    "+ 60"]

dataset_CHP_PS["FAIXA_ETARIA"] = np.select(condicoes, faixas, default="Sem faixa")

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

df_CHP = dataset_CHP_PS_0.groupby(['ANO', 'UNIDADE', 'MUNICIPIO','CURSO', 'TURNO','FAIXA_ETARIA'],as_index=False).agg({'MATRICULAS':'sum'})
df_CHP.columns = df_CHP.columns.str.strip()

ano_ref = 2025
df_CHP_Pagante_menor_ = df_CHP[df_CHP['ANO'] <= ano_ref]
df_CHP_Pagante_igual_ = df_CHP_Pagante_menor_.copy(deep=True)
df_CHP_Pagante_igual_['CURSO'] = df_CHP_Pagante_igual_['CURSO'].str.upper()
df_CHP_Pagante_igual_['CURSO'] = (df_CHP_Pagante_igual_['CURSO'].str.replace(r'^TÉCNICO\s+EM\s+', '', regex=True, flags=re.IGNORECASE).str.strip().str.upper())
df_CHP_Pagante_igual_['UNIDADE'] = df_CHP_Pagante_igual_['UNIDADE'].replace({'LEM':'LUÍS EDUARDO MAGALHÃES','FEIRA':'FEIRA DE SANTANA','CONQUISTA':'VITÓRIA DA CONQUISTA'})


####################################################################################################
# >>>>>>> - FONTE CURSOS TÉCNICOS SENAI [ITINERARIO - CBO] - <<<<<<< #
flle_iti = r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\dados_extraidos_final.xlsx"
itinerario = pd.read_excel(flle_iti, sheet_name='Sheet1')
itinerario['MODALIDADE'] = np.where(itinerario['Nome do Item'].str.contains('Aperfeiçoamento|Especialista|Especialização', case=False, na=False), 'CAEP', 'CHP')
itinerario['Nome do Item'] = itinerario['Nome do Item'].str.upper()
itinerario['Nome do Item'] = itinerario['Nome do Item'].replace({'TÉCNICO EM ':'',
                                                             'TECNÓLOGO EM ':''}, regex=True)
itinerario_1 = itinerario[itinerario['MODALIDADE'] == 'CHP'].drop_duplicates().reset_index(drop=True)


##################################################+ # + # + # + # + # + # + # + # + # + # + # + # + #
##### >>>>>>> ETL <<<<<<< #####
##################################################

# >>>>>>> - CHP PAGANTE - <<<<<<< #
df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_.copy(deep=True)
df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_2.groupby(['ANO','CURSO','UNIDADE','MUNICIPIO'], as_index=False).agg({'MATRICULAS':'sum'})

# >>>>>>> - RAIS ESTABELECIMENTO - <<<<<<< #
df_CHP_Pagante_igual_2= df_CHP_Pagante_igual_2.merge(df_rais_ESTAB_2[['ANO','MUNICIPIO','QTD_EMPRESAS']],
                                                           how='left',
                                                           left_on=['MUNICIPIO','ANO'],
                                                           right_on=['MUNICIPIO','ANO'])

df_CHP_Pagante_igual_2 = df_CHP_Pagante_igual_2.rename(columns={'MATRICULAS':'MAT_PAG'})

# >>>>>>> - ITINERÁRIO - <<<<<<< #
join_CHP_itinerario_2025_1 = df_CHP_Pagante_igual_2.merge(itinerario_1[['Nome do Item','CBO']], 
                                                          how='left',
                                                          left_on='CURSO', 
                                                          right_on='Nome do Item')

join_CHP_itinerario_2025_1.drop(columns='Nome do Item', inplace=True)
join_CHP_itinerario_2025_1['CBO_familia'] = join_CHP_itinerario_2025_1['CBO'].str.split('-').str[0]
join_CHP_itinerario_2025_1['CBO'] = join_CHP_itinerario_2025_1['CBO'].str.replace('-','',regex=False)
join_CHP_itinerario_2025_1['CBO_MUNICIPIO'] = join_CHP_itinerario_2025_1['CBO'] + join_CHP_itinerario_2025_1['MUNICIPIO']

# >>>>>>> - CAGED - <<<<<<< #
join_CHP_itinerario_2025_2_caged = join_CHP_itinerario_2025_1.merge(
    df_caged_2[['ANO','CBO_MUNICIPIO',
                'SALARIO_ADMITIDOS','SALARIO_DESLIGADOS','PRESSAO_SALARIAL',
                'SUM_ADMITIDOS','SUM_DESLIGADOS','SALDO_EMPREGO']],
                    how='left', 
                    left_on=['CBO_MUNICIPIO','ANO'], 
                    right_on=['CBO_MUNICIPIO','ANO'])

# >>>>>>> - RAIS VINCULOS - <<<<<<< #
join_CHP_itinerario_2025_3_caged= join_CHP_itinerario_2025_2_caged.merge(df_rais_VINC_2[['ANO','CBO_MUNICIPIO','QTD_VINCULOS','SALARIO_MEDIO']],
                                                           how='left',
                                                           left_on=['CBO_MUNICIPIO','ANO'],
                                                           right_on=['CBO_MUNICIPIO','ANO'])

# >>>>>>> - CONCORRÊNCIA INEP - <<<<<<< #
join_CHP_itinerario_2025_4_caged = join_CHP_itinerario_2025_3_caged.merge(df_curso_tec_inep_2[['ANO','MUNICIPIO','QTD_CONC_INEP','QTD_MAT_CONC']],
                                                           how='left',
                                                           left_on=['MUNICIPIO','ANO'],
                                                           right_on=['MUNICIPIO','ANO'])

# >>>>>>> - DATAFRAME FINAL - <<<<<<< #
join_CHP_itinerario_2025_5_caged = join_CHP_itinerario_2025_4_caged[['ANO','CURSO', 'UNIDADE', 'MUNICIPIO','MAT_PAG',
                                                                     'QTD_CONC_INEP','QTD_MAT_CONC',
                                                                     'QTD_EMPRESAS', 'QTD_VINCULOS','SUM_ADMITIDOS',
                                                                     'SUM_DESLIGADOS','SALDO_EMPREGO','SALARIO_MEDIO',
                                                                     'SALARIO_ADMITIDOS','SALARIO_DESLIGADOS','PRESSAO_SALARIAL'
                                                                     ]].copy(deep=True)

join_CHP_itinerario_2025_6_caged = join_CHP_itinerario_2025_5_caged.copy(deep=True)
join_CHP_itinerario_2025_6_caged['PRESSAO_SALARIAL'] = join_CHP_itinerario_2025_6_caged['PRESSAO_SALARIAL'].fillna(0).astype(float)
join_CHP_itinerario_2025_6_caged = join_CHP_itinerario_2025_6_caged[(join_CHP_itinerario_2025_6_caged['SALDO_EMPREGO'].notnull()) & (join_CHP_itinerario_2025_6_caged['QTD_VINCULOS'].notnull())].reset_index(drop=True)

labels_faixas = ['Abaixo ou igual a 20', 'Entre 21 e 40', 'Acima ou igual a 41']
join_CHP_itinerario_2025_6_caged['FAIXA_MAT'] = pd.cut(join_CHP_itinerario_2025_6_caged['MAT_PAG'],
                                                       bins=[-np.inf, 20, 40, np.inf],
                                                       labels=labels_faixas,
                                                       right=True)

cols_num = join_CHP_itinerario_2025_6_caged.select_dtypes(include='number').columns
join_CHP_itinerario_2025_6_caged[cols_num] = (join_CHP_itinerario_2025_6_caged[cols_num].fillna(0))

join_CHP_itinerario_2025_6_caged.to_pickle(r"C:\Users\anderson.pereira\OneDrive - Sistema FIEB\Inteligência de Mercado\Data Science\DeploymentML\Streamlit_dsim_ML\check_dataset_0.pkl")
print("Dataset final processado e salvo como check_dataset_0.pkl")

