<center>

# Análise das Redes de Comércio Internacional de Máscaras Cirúrgicas

### O Impacto da Pandemia COVID-19 nas Dinâmicas Comerciais Globais (2015–2024)

---

**Autor:** Ricardo Vicente  
**Mestrado:** Análise de Dados e Sistemas de Apoio à Decisão  
**UC:** Otimização em Redes e Redes Sociais  
**Data:** Junho de 2025

---

</center>


# Sumário

**1. Preparação dos dados** <br>
&ensp;*1.1 - Imports e carregamento de dados* <br>
&ensp;*1.2 - Informações básicas dos dados* <br>
&ensp;*1.3 - Info sobre Países* <br>
&ensp;*1.4 - Homogeneizar e corrigir ISO de países* <br>
&ensp;*1.5 - Limpeza de dados e Est. Descritivas* <br>
&ensp;*1.6 - Análise básica de trocas* <br>
**2. AED** <br>
&ensp;*2.1 - Variações anuais* <br>
&ensp;*2.2 - Top Exportadores em $* <br>
&ensp;*2.3 - Top Exportadores em qty* <br>
&ensp;*2.4 - Modificações nos Tops* <br>
**3. Construção das redes** <br>
&ensp;*3.1 - Rede Pré-Pandemia - Spring Layout* <br>
&ensp;*3.2 - Análise por Períodos* <br>
&emsp;3.2.1 - Estatísticas básicas <br>
&emsp;3.2.2 - Medidas de Centralidade <br>
&emsp;3.2.3 - Propriedades de conectividade <br>
&emsp;3.2.4 - Medidas de distância <br>
&emsp;3.2.5 - Detecção de Comunidades <br>
&ensp;*3.3 - Mapas Temporais* <br>
**4. O Caso de Portugal** <br>

# 1. Preparação dos Dados

## 1.1 Imports e carregamento de dados


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import random
import pycountry
import math
from IPython.display import display
import country_converter as coco
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import community  
from networkx.algorithms import community as nx_community
import plotly.graph_objects as go

#settings de vis
plt.rcParams['figure.figsize'] = [10, 6]
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# carregar dados
df = pd.read_csv('dataset4.csv', sep=";", encoding='latin-1')

display(df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>refYear</th>
      <th>refMonth</th>
      <th>flowDesc</th>
      <th>reporterISO</th>
      <th>partnerISO</th>
      <th>qty</th>
      <th>primaryValue</th>
      <th>unitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>1</td>
      <td>Import</td>
      <td>DZA</td>
      <td>W00</td>
      <td>0</td>
      <td>391716,54</td>
      <td>#DIV/0!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1</td>
      <td>Import</td>
      <td>DZA</td>
      <td>CAN</td>
      <td>0</td>
      <td>466,05</td>
      <td>#DIV/0!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>1</td>
      <td>Import</td>
      <td>DZA</td>
      <td>CHN</td>
      <td>0</td>
      <td>154815</td>
      <td>#DIV/0!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>1</td>
      <td>Import</td>
      <td>DZA</td>
      <td>CZE</td>
      <td>0</td>
      <td>54,72</td>
      <td>#DIV/0!</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>1</td>
      <td>Import</td>
      <td>DZA</td>
      <td>FRA</td>
      <td>0</td>
      <td>47176,61</td>
      <td>#DIV/0!</td>
    </tr>
  </tbody>
</table>
</div>


## 1.2 Informações básicas dos dados


```python
# colunas no df
print("Colunas no DataFrame:\n")
print(df.columns.tolist())
print("\n",100*"=")

# infos básicas do dataset
print(f"\nTotal de registos: {len(df)}\n")
print(f"Anos incluídos: {df['refYear'].unique()}")
print(f"Número de países Exportadores: {df['reporterISO'].nunique()}")
print(f"Número de países Importadores: {df['partnerISO'].nunique()}")

print("\n",100*"=")

print("\nTipos de dados:\n")
display(df.dtypes)
```

    Colunas no DataFrame:
    
    ['refYear', 'refMonth', 'flowDesc', 'reporterISO', 'partnerISO', 'qty', 'primaryValue', 'unitPrice']
    
     ====================================================================================================
    
    Total de registos: 794265
    
    Anos incluídos: [2015 2016 2017 2018 2019 2020 2021 2022 2023 2024]
    Número de países Exportadores: 156
    Número de países Importadores: 247
    
     ====================================================================================================
    
    Tipos de dados:
    
    


    refYear          int64
    refMonth         int64
    flowDesc        object
    reporterISO     object
    partnerISO      object
    qty             object
    primaryValue    object
    unitPrice       object
    dtype: object


## 1.3 Info sobre Países


```python
print(f"Países Exportadores: {df['reporterISO'].unique()}\n")
print(f"\nPaíses Importadores: {df['partnerISO'].unique()}")
```

    Países Exportadores: ['DZA' 'AND' 'AGO' 'ATG' 'AZE' 'ARG' 'AUS' 'AUT' 'BHR' 'ARM' 'BRB' 'BEL'
     'BMU' 'BOL' 'BIH' 'BWA' 'BRA' 'BLZ' 'BRN' 'BGR' 'MMR' 'BDI' 'BLR' 'KHM'
     'CAN' 'CPV' 'CHL' 'COL' 'COM' 'COG' 'COD' 'HRV' 'CYP' 'CZE' 'BEN' 'DNK'
     'DOM' 'ECU' 'SLV' 'ETH' 'EST' 'FJI' 'FIN' 'FRA' 'PYF' 'GEO' 'DEU' 'GHA'
     'GRC' 'GRL' 'GRD' 'GTM' 'GUY' 'HKG' 'HUN' 'ISL' 'IDN' 'IRL' 'ISR' 'ITA'
     'CIV' 'JPN' 'KAZ' 'KOR' 'KWT' 'KGZ' 'LAO' 'LSO' 'LVA' 'LTU' 'LUX' 'MAC'
     'MDG' 'MWI' 'MYS' 'MLT' 'MUS' 'MEX' 'MDA' 'MNE' 'MSR' 'MAR' 'MOZ' 'OMN'
     'NLD' 'NCL' 'NZL' 'NIC' 'NOR' 'PLW' 'PAK' 'PAN' 'PRY' 'PER' 'PHL' 'POL'
     'PRT' 'QAT' 'ROU' 'RUS' 'RWA' 'KNA' 'VCT' 'STP' 'SAU' 'SEN' 'SRB' 'SYC'
     'IND' 'SGP' 'SVK' 'VNM' 'SVN' 'ZAF' 'ZWE' 'ESP' 'SDN' 'SWZ' 'SWE' 'CHE'
     'THA' 'TTO' 'TUR' 'UGA' 'UKR' 'MKD' 'EGY' 'GBR' 'TZA' 'USA' 'BFA' 'URY'
     'WSM' 'YEM' 'ZMB' 'TGO' 'GMB' 'SLB' 'CHN' 'HND' 'MNG' 'NER' 'SLE' 'PSE'
     'KIR' 'JOR' 'KEN' 'NAM' 'ARE' 'CRI' 'MDV' 'NGA' 'ABW' 'UZB' 'MRT' 'TUN']
    
    
    Países Importadores: ['W00' 'CAN' 'CHN' 'CZE' 'FRA' 'ITA' 'KOR' 'S19' 'PAK' 'POL' 'PRT' 'RUS'
     'IND' 'ESP' 'SWE' 'TUN' 'TUR' 'EGY' 'GBR' 'USA' 'DEU' 'NLD' 'CHE' 'BEL'
     'BRA' 'COD' 'DNK' 'ISR' 'JPN' 'LBN' 'MEX' 'MAR' 'NAM' 'SGP' 'ZAF' 'ARE'
     'SXM' 'PER' 'AUS' 'IRN' 'LKA' 'COL' 'HKG' 'MYS' 'ROU' 'VNM' 'URY' 'AUT'
     'BGD' 'BIH' 'KHM' 'FJI' 'FIN' 'HND' 'HUN' 'IDN' 'IRL' 'KEN' 'LVA' 'MDG'
     'NPL' 'NZL' 'NOR' 'PHL' 'SVK' 'THA' 'UKR' 'MKD' 'BOL' 'HRV' 'EST' 'GMB'
     'LUX' 'SRB' 'SVN' '_X ' 'CRI' 'GIN' 'JAM' 'PAN' 'MAF' 'SUR' 'CHL' 'SAU'
     'ARG' 'GTM' 'BGR' 'ZMB' 'VGB' 'ECU' 'BRB' 'GRC' 'UGA' 'LTU' 'TZA' 'DMA'
     'NIC' 'PRY' 'ATA' 'GHA' 'KWT' 'MLI' 'NCL' 'PNG' 'SEN' 'TON' 'X1 ' 'AZE'
     'BHS' 'ARM' 'BLR' 'CMR' 'CYP' 'GAB' 'GEO' 'ISL' 'KAZ' 'LBR' 'LBY' 'MLT'
     'MNE' 'OMN' 'QAT' 'SYR' 'X2 ' 'CYM' 'GRD' 'LCA' 'VCT' 'TTO' 'DZA' 'BEN'
     'GNQ' 'ERI' 'PYF' 'GIB' 'IRQ' 'JOR' 'MUS' 'F19' 'SLE' 'AGO' 'CIV' 'MOZ'
     'AFG' 'DOM' 'TKM' 'SLV' 'E19' 'SMR' 'LAO' 'MDA' 'NGA' 'ALB' 'BHR' 'BMU'
     'CUB' 'CUW' 'UMI' 'SPM' 'BFA' 'HTI' 'VEN' 'MDV' 'FRO' 'GRL' 'KGZ' 'UZB'
     'SLB' 'VUT' 'ATG' 'BVT' 'PRK' 'GUM' 'AND' 'CAF' 'TCD' 'COM' 'COG' 'DJI'
     'MRT' 'ABW' 'RWA' 'BLM' 'SYC' 'ZWE' 'TGO' 'PSE' 'MAC' 'BRN' 'GUY' 'MNG'
     'SDN' 'YEM' 'MMR' 'BLZ' 'ETH' 'BDI' 'WLF' 'COK' 'PLW' 'WSM' 'SOM' 'CPV'
     'GNB' 'STP' 'LSO' 'XX ' 'BTN' 'MWI' 'NER' 'MNP' 'BWA' 'SWZ' 'TJK' 'MHL'
     'KNA' 'TCA' 'KIR' 'BES' 'A79' 'NIU' 'TLS' 'AIA' 'NRU' 'TKL' 'O19' 'MSR'
     'ASM' 'TUV' 'FSM' 'SSD' 'CXR' 'NFK' 'IOT' 'MYT' 'CCK' 'SHN' 'FLK' 'SGS'
     'VAT' 'ATF' 'PCN' 'ESH' 'ATB' 'A59' 'HMD']
    

### 1.4 Homogeneizar e corrigir ISO de países


```python
# Todo este bloco de código foi sugerido pelo chatGPT. Utilizei para, dada a lista completa de países, 
# detectar redundâncias, erros ou nomes que não correspondem a países reais. Após fazer isso, sugeriu esta função para corrigir
# nomes. A função estava mais simples que a minha, optei usar esta. 

def corrigir_nomes_paises(df):
    
    df_corrigido = df.copy()
      
    # Códigos que não existem 
    entidades_especiais = ['W00', 'S19', '_X ', 'X1 ', 'X2 ', 'F19', 'E19', 'A79', 'O19', 'A59', 'XX ']
    
    # Filtrar apenas países reais (removendo todas as entidades especiais)

    df_corrigido = df_corrigido[
        (~df_corrigido['reporterISO'].isin(entidades_especiais)) & 
        (~df_corrigido['partnerISO'].isin(entidades_especiais))
    ]
    
    return df_corrigido

df_corrigido = corrigir_nomes_paises(df)
```


```python
# infos básicas do dataset
print(f"\nTotal de registos: {len(df_corrigido)}\n")
print(f"Anos incluídos: {df_corrigido['refYear'].unique()}")
print(f"Número de países Exportadores: {df_corrigido['reporterISO'].nunique()}")
print(f"Número de países Importadores: {df_corrigido['partnerISO'].nunique()}")
```

    
    Total de registos: 748813
    
    Anos incluídos: [2015 2016 2017 2018 2020 2021 2022 2023 2024]
    Número de países Exportadores: 155
    Número de países Importadores: 236
    

**Nota:** Ao retirar o WOO como _"entidade_especial" o ano de 2019 é todo eliminado pois todas os importadores de 2019 estão listados como WOO em partnerISO

## 1.5 Limpeza de dados e Est. Descritivas


```python
# limpar #N/A decorrente de qty = 0 no cálculo do unitPrice - está com #N/A porque fiz este cálculo no excel, antes de carregar os dados aqui
df_corrigido['qty'] = df_corrigido['qty'].replace('#N/A', np.nan)

# conversão de formatos numéricos
df_corrigido['primaryValue'] = df_corrigido['primaryValue'].astype(str).str.replace(',', '.', regex=False)
df_corrigido['qty'] = df_corrigido['qty'].astype(str).str.replace(',', '.', regex=False)
df_corrigido['primaryValue'] = pd.to_numeric(df_corrigido['primaryValue'], errors='coerce')
df_corrigido['qty'] = pd.to_numeric(df_corrigido['qty'], errors='coerce')

print("\nValores ausentes por coluna:\n")
print(df_corrigido.isnull().sum())

# criar colunas para países, há nomes que precisam de ser homogeneizados 
df_corrigido['Exportador'] = df_corrigido['reporterISO']
df_corrigido['Importador'] = df_corrigido['partnerISO']

# criar df filtrado apenas com exportações
print("Valores únicos na coluna flowDesc:")
print(df_corrigido['flowDesc'].unique())

df_exports = df_corrigido[df_corrigido['flowDesc'] == 'Export'].copy()
print(f"\nTotal de registos de exportação: {len(df_exports)}\n")

# verificar valores ausentes
print("\nValores ausentes após conversão:\n")
print(df_exports[['primaryValue', 'qty']].isna().sum())

# resumo
print("\nResumo estatístico após limpeza:")
stats = df_exports[['primaryValue', 'qty']].describe().round(2)
html_stats = stats.style.format("{:,.2f}")
display(html_stats)
```

    
    Valores ausentes por coluna:
    
    refYear         0
    refMonth        0
    flowDesc        0
    reporterISO     0
    partnerISO      0
    qty             0
    primaryValue    0
    unitPrice       0
    dtype: int64
    Valores únicos na coluna flowDesc:
    ['Import' 'Export']
    
    Total de registos de exportação: 349622
    
    
    Valores ausentes após conversão:
    
    primaryValue    0
    qty             0
    dtype: int64
    
    Resumo estatístico após limpeza:
    


<style type="text/css">
</style>
<table id="T_237c3">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_237c3_level0_col0" class="col_heading level0 col0" >primaryValue</th>
      <th id="T_237c3_level0_col1" class="col_heading level0 col1" >qty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_237c3_level0_row0" class="row_heading level0 row0" >count</th>
      <td id="T_237c3_row0_col0" class="data row0 col0" >349,622.00</td>
      <td id="T_237c3_row0_col1" class="data row0 col1" >349,622.00</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row1" class="row_heading level0 row1" >mean</th>
      <td id="T_237c3_row1_col0" class="data row1 col0" >472,476.06</td>
      <td id="T_237c3_row1_col1" class="data row1 col1" >34,380.18</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row2" class="row_heading level0 row2" >std</th>
      <td id="T_237c3_row2_col0" class="data row2 col0" >12,406,529.79</td>
      <td id="T_237c3_row2_col1" class="data row2 col1" >577,240.69</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row3" class="row_heading level0 row3" >min</th>
      <td id="T_237c3_row3_col0" class="data row3 col0" >0.00</td>
      <td id="T_237c3_row3_col1" class="data row3 col1" >0.00</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row4" class="row_heading level0 row4" >25%</th>
      <td id="T_237c3_row4_col0" class="data row4 col0" >869.00</td>
      <td id="T_237c3_row4_col1" class="data row4 col1" >8.90</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row5" class="row_heading level0 row5" >50%</th>
      <td id="T_237c3_row5_col0" class="data row5 col0" >7,749.73</td>
      <td id="T_237c3_row5_col1" class="data row5 col1" >200.00</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row6" class="row_heading level0 row6" >75%</th>
      <td id="T_237c3_row6_col0" class="data row6 col0" >62,773.85</td>
      <td id="T_237c3_row6_col1" class="data row6 col1" >2,870.99</td>
    </tr>
    <tr>
      <th id="T_237c3_level0_row7" class="row_heading level0 row7" >max</th>
      <td id="T_237c3_row7_col0" class="data row7 col0" >3,653,069,488.00</td>
      <td id="T_237c3_row7_col1" class="data row7 col1" >77,478,428.00</td>
    </tr>
  </tbody>
</table>



## 1.6 Análise básica de trocas


```python
# versão agregada por ano (soma dos valores)
df_export_annual = df_exports.groupby(['refYear', 'reporterISO', 'partnerISO'])['primaryValue'].sum().reset_index()

# dados de exportação agregados
print("\nDados de exportação agregados por ano:")
display(df_export_annual.head())

# principais países exportadores e importadores
top_exporters = df_export_annual.groupby('reporterISO')['primaryValue'].sum().sort_values(ascending=False).head(10)
top_importers = df_export_annual.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(10)

print("\nTop 10 países exportadores:")

display(top_exporters)

print("\nTop 10 países importadores:")
display(top_importers)
```

    
    Dados de exportação agregados por ano:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>refYear</th>
      <th>reporterISO</th>
      <th>partnerISO</th>
      <th>primaryValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>AGO</td>
      <td>ARE</td>
      <td>188.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>AGO</td>
      <td>BEL</td>
      <td>3347.41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>AGO</td>
      <td>BRA</td>
      <td>49.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>AGO</td>
      <td>COD</td>
      <td>157.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>AGO</td>
      <td>COG</td>
      <td>45907.17</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 países exportadores:
    


    reporterISO
    CHN    9.582094e+10
    DEU    9.595403e+09
    USA    7.148129e+09
    MEX    5.745379e+09
    VNM    5.534274e+09
    IND    3.760615e+09
    NLD    3.366292e+09
    POL    2.576971e+09
    FRA    2.408748e+09
    TUR    2.206861e+09
    Name: primaryValue, dtype: float64


    
    Top 10 países importadores:
    


    partnerISO
    USA    4.319084e+10
    DEU    1.379284e+10
    JPN    1.092044e+10
    FRA    9.253455e+09
    GBR    8.129058e+09
    ITA    5.287905e+09
    CAN    5.286796e+09
    ESP    4.798290e+09
    NLD    4.716897e+09
    MEX    3.948564e+09
    Name: primaryValue, dtype: float64


# 2. AED

## 2.1 Variações anuais


```python
# análise temporal do valor total de exportações por ano
yearly_stats = df_exports.groupby('refYear').agg({
    'primaryValue': 'sum',
    'qty': 'sum',
    'Exportador': 'nunique',
    'Importador': 'nunique'
}).reset_index()

# preço médio por unidade para cada ano
yearly_stats['Preço Médio por Unidade (USD)'] = yearly_stats['primaryValue'] / yearly_stats['qty']

# renomear colunas
yearly_stats = yearly_stats.rename(columns={
    'primaryValue': 'Valor Total (USD)',
    'qty': 'Quantidade Total',
    'Exportador': 'Número de Exportadores',
    'Importador': 'Número de Importadores'
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# mil milhões = 1e9 
# Grafico 1: Valor Total das Exportações
axes[0, 0].bar(yearly_stats['refYear'], yearly_stats['Valor Total (USD)'] / 1e9, color='steelblue')
axes[0, 0].set_title('Valor Total das Exportações de Máscaras')
axes[0, 0].set_xlabel('Ano')
axes[0, 0].set_ylabel('mil Milhões USD')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
# Adicionar rótulos de valor
for i, value in enumerate(yearly_stats['Valor Total (USD)']):
    axes[0, 0].text(yearly_stats['refYear'].iloc[i], value/1e9 + 2, f'${value/1e9:.1f}mM', 
                 ha='center', va='bottom')

# Grafico 2: Quantidade Total de Máscaras
axes[0, 1].bar(yearly_stats['refYear'], yearly_stats['Quantidade Total'] / 1e9, color='darkgreen')
axes[0, 1].set_title('Quantidade Total de Máscaras Exportadas')
axes[0, 1].set_xlabel('Ano')
axes[0, 1].set_ylabel('mil Milhões de Unidades')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
# Adicionar rótulos de valor
for i, value in enumerate(yearly_stats['Quantidade Total']):
    axes[0, 1].text(yearly_stats['refYear'].iloc[i], value/1e9 + 0.2, f'{value/1e9:.1f}mM', 
                 ha='center', va='bottom')

# Grafico 3: Preço Médio por Unidade
axes[1, 0].bar(yearly_stats['refYear'], yearly_stats['Preço Médio por Unidade (USD)'], color='purple')
axes[1, 0].set_title('Preço Médio por Unidade de Máscara')
axes[1, 0].set_xlabel('Ano')
axes[1, 0].set_ylabel('USD')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
# Adicionar rótulos de valor
for i, value in enumerate(yearly_stats['Preço Médio por Unidade (USD)']):
    axes[1, 0].text(yearly_stats['refYear'].iloc[i], value + 1, f'${value:.2f}', 
                 ha='center', va='bottom')

# Grafico 4: Número de Países Exportadores 
axes[1, 1].plot(yearly_stats['refYear'], yearly_stats['Número de Exportadores'], 'o-', color='blue', label='Exportadores')

importers_adjusted = yearly_stats['Número de Importadores'].copy()
importers_adjusted.loc[yearly_stats['refYear'] == 2019]  # Marcar 2019 como dados ausentes

mask = yearly_stats['refYear'] != 2019
axes[1, 1].plot(yearly_stats.loc[mask, 'refYear'], importers_adjusted[mask], 's-', color='red', label='Importadores')
axes[1, 1].set_title('Número de Países Envolvidos no Comércio')
axes[1, 1].set_xlabel('Ano')
axes[1, 1].set_ylabel('Número de Países')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
# adicionar nota sobre o ano 2019
axes[1, 1].annotate('Dados de 2019 apenas\nagregados como "World"', 
                  xy=(2019, 100), xytext=(2019, 150),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                  ha='center', fontsize=10)

plt.tight_layout()
plt.show()


print("\nComparação de Períodos:")
# tirar 2019 da análise pré-pandemia devido à limitação dos dados
pre_pandemia_ajustado = yearly_stats[yearly_stats['refYear'] < 2019]
durante_pandemia = yearly_stats[(yearly_stats['refYear'] >= 2020) & (yearly_stats['refYear'] <= 2021)]
pos_pandemia = yearly_stats[yearly_stats['refYear'] > 2021]

print(f"Média de Valor Total Pré-Pandemia (2015-2019): ${pre_pandemia_ajustado['Valor Total (USD)'].mean()/1e9:.2f} mil Milhões")
print(f"Média de Valor Total Durante Pandemia (2020-2021): ${durante_pandemia['Valor Total (USD)'].mean()/1e9:.2f} mil Milhões")
print(f"Média de Valor Total Pós-Pandemia (2022-2024): ${pos_pandemia['Valor Total (USD)'].mean()/1e9:.2f} mil Milhões")

print(f"\nMédia de Preço por Unidade Pré-Pandemia (2015-2019): ${pre_pandemia_ajustado['Preço Médio por Unidade (USD)'].mean():.2f}")
print(f"Média de Preço por Unidade Durante Pandemia (2020-2021): ${durante_pandemia['Preço Médio por Unidade (USD)'].mean():.2f}")
print(f"Média de Preço por Unidade Pós-Pandemia (2022-2024): ${pos_pandemia['Preço Médio por Unidade (USD)'].mean():.2f}")
```


    
![png](output_20_0.png)
    


    
    Comparação de Períodos:
    Média de Valor Total Pré-Pandemia (2015-2019): $9.23 mil Milhões
    Média de Valor Total Durante Pandemia (2020-2021): $45.43 mil Milhões
    Média de Valor Total Pós-Pandemia (2022-2024): $12.47 mil Milhões
    
    Média de Preço por Unidade Pré-Pandemia (2015-2019): $11.41
    Média de Preço por Unidade Durante Pandemia (2020-2021): $18.72
    Média de Preço por Unidade Pós-Pandemia (2022-2024): $9.78
    

## 2.2 Top Exportadores em $


```python
periodos = {
    "Pré-Pandemia (2015-2018)": df_exports[df_exports['refYear'] < 2019],
    "Durante a Pandemia (2020-2021)": df_exports[(df_exports['refYear'] >= 2020) & (df_exports['refYear'] <= 2021)],
    "Pós-Pandemia (2022-2024)": df_exports[df_exports['refYear'] > 2021]
}

#função para top exporters
def top_exportadores(df, n=10):
    return (
        df.groupby('Exportador')['primaryValue'].sum()
        .nlargest(n)
        .reset_index()
    )


def plot_top(df_top, titulo):
    df_top = df_top.sort_values('primaryValue')
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_top['Exportador'], df_top['primaryValue'] / 1e9, color='steelblue')
    plt.xlabel('Exportações (mil Milhões USD)')
    plt.title(titulo)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}mM', va='center')
    plt.tight_layout()
    plt.show()

# amrazenar resultados e mostrar
tops = {}
for nome, dados in periodos.items():
    tops[nome] = top_exportadores(dados)
    print(f"\nTop 10 Exportadores - {nome}")
    for i, row in tops[nome].iterrows():
        print(f"{i+1}. {row['Exportador']}: ${row['primaryValue'] / 1e9:.2f}mM")
    plot_top(tops[nome], f'Top 10 Exportadores de Máscaras - {nome}')
```

    
    Top 10 Exportadores - Pré-Pandemia (2015-2018)
    1. CHN: $14.06mM
    2. DEU: $3.05mM
    3. USA: $2.66mM
    4. VNM: $2.09mM
    5. MEX: $2.02mM
    6. IND: $2.00mM
    7. NLD: $0.98mM
    8. FRA: $0.74mM
    9. GBR: $0.68mM
    10. HKG: $0.67mM
    


    
![png](output_22_1.png)
    


    
    Top 10 Exportadores - Durante a Pandemia (2020-2021)
    1. CHN: $66.26mM
    2. DEU: $3.42mM
    3. VNM: $2.49mM
    4. USA: $2.11mM
    5. MEX: $1.48mM
    6. NLD: $1.23mM
    7. HKG: $1.00mM
    8. KOR: $0.99mM
    9. TUR: $0.97mM
    10. FRA: $0.92mM
    


    
![png](output_22_3.png)
    


    
    Top 10 Exportadores - Pós-Pandemia (2022-2024)
    1. CHN: $15.50mM
    2. DEU: $3.13mM
    3. USA: $2.38mM
    4. MEX: $2.25mM
    5. NLD: $1.16mM
    6. POL: $1.12mM
    7. IND: $1.04mM
    8. VNM: $0.95mM
    9. TUR: $0.79mM
    10. FRA: $0.76mM
    


    
![png](output_22_5.png)
    


## 2.3 Top Exportadores em qty


```python
periodos = {
    "Pré-Pandemia (2015-2018)": df_exports[df_exports['refYear'] < 2019],
    "Durante a Pandemia (2020-2021)": df_exports[(df_exports['refYear'] >= 2020) & (df_exports['refYear'] <= 2021)],
    "Pós-Pandemia (2022-2024)": df_exports[df_exports['refYear'] > 2021]
}

#função para top exporters
def top_exportadores(df, n=10):
    return (
        df.groupby('Exportador')['qty'].sum()
        .nlargest(n)
        .reset_index()
    )

def plot_top(df_top, titulo):
    df_top = df_top.copy()
    df_top['qty'] = df_top['qty'] / 1000
    df_top = df_top.sort_values('qty')

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_top['Exportador'], df_top['qty'], color='steelblue')
    plt.xlabel('Exportações (toneladas)')
    plt.title(titulo)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f} t', va='center')
    plt.tight_layout()
    plt.show()

# armazenar resultados e mostrar
tops = {}
for nome, dados in periodos.items():
    tops[nome] = top_exportadores(dados)
    print(f"\nTop 10 Exportadores - {nome}")
    for i, row in tops[nome].iterrows():
        print(f"{i+1}. {row['Exportador']}: {row['qty'] / 1000:.2f} toneladas")
    plot_top(tops[nome], f'Top 10 Exportadores de Máscaras - {nome}')

```

    
    Top 10 Exportadores - Pré-Pandemia (2015-2018)
    1. CHN: 1370895.94 toneladas
    2. HKG: 771349.28 toneladas
    3. DEU: 207818.22 toneladas
    4. NLD: 126099.65 toneladas
    5. VNM: 116922.12 toneladas
    6. MEX: 112478.94 toneladas
    7. IND: 107670.14 toneladas
    8. TUR: 55077.03 toneladas
    9. USA: 47960.28 toneladas
    10. THA: 33598.85 toneladas
    


    
![png](output_24_1.png)
    


    
    Top 10 Exportadores - Durante a Pandemia (2020-2021)
    1. CHN: 3302422.32 toneladas
    2. DEU: 165005.51 toneladas
    3. USA: 160471.86 toneladas
    4. NLD: 123349.83 toneladas
    5. VNM: 100976.72 toneladas
    6. TUR: 77638.38 toneladas
    7. POL: 64099.71 toneladas
    8. IND: 60624.55 toneladas
    9. MEX: 59482.16 toneladas
    10. HKG: 44607.69 toneladas
    


    
![png](output_24_3.png)
    


    
    Top 10 Exportadores - Pós-Pandemia (2022-2024)
    1. CHN: 2119386.72 toneladas
    2. NLD: 236143.53 toneladas
    3. USA: 200440.61 toneladas
    4. DEU: 188046.24 toneladas
    5. MEX: 132135.50 toneladas
    6. TUR: 101381.22 toneladas
    7. IND: 98644.41 toneladas
    8. POL: 85724.99 toneladas
    9. VNM: 61756.26 toneladas
    10. ESP: 53605.98 toneladas
    


    
![png](output_24_5.png)
    


## 2.4 Modificações nos Tops


```python
# comparar rankings entre períodos
def comparar_rankings(top_a, top_b, nome_a, nome_b):
    set_a, set_b = set(top_a['Exportador']), set(top_b['Exportador'])
    novos = set_b - set_a
    sairam = set_a - set_b
    comuns = set_a & set_b

    print(f"\n Mudanças do Top 10 entre {nome_a} e {nome_b}:")
    if novos:
        print(f" Entraram no top 10 em {nome_b}: {', '.join(novos)}")
    if sairam:
        print(f" Saíram do top 10 em {nome_b}: {', '.join(sairam)}")
    
    # análise de variação de posição
    rank_a = {c: i+1 for i, c in enumerate(top_a['Exportador'])}
    rank_b = {c: i+1 for i, c in enumerate(top_b['Exportador'])}
    mudancas = [(c, rank_a[c], rank_b[c], rank_a[c] - rank_b[c]) for c in comuns]
    mudancas.sort(key=lambda x: abs(x[3]), reverse=True)

    print("\n Maiores mudanças de posição:")
    for c, r1, r2, delta in mudancas[:5]:
        direcao = "subiu" if delta > 0 else "desceu"
        print(f"{c}: {r1}º -> {r2}º ({direcao} {abs(delta)} posições)")

comparar_rankings(tops["Pré-Pandemia (2015-2018)"], tops["Durante a Pandemia (2020-2021)"],
                  "Pré-Pandemia", "Durante a Pandemia")
comparar_rankings(tops["Durante a Pandemia (2020-2021)"], tops["Pós-Pandemia (2022-2024)"],
                  "Durante a Pandemia", "Pós-Pandemia")

```

    
     Mudanças do Top 10 entre Pré-Pandemia e Durante a Pandemia:
     Entraram no top 10 em Durante a Pandemia: POL
     Saíram do top 10 em Durante a Pandemia: THA
    
     Maiores mudanças de posição:
    HKG: 2º -> 10º (desceu 8 posições)
    USA: 9º -> 3º (subiu 6 posições)
    MEX: 6º -> 9º (desceu 3 posições)
    TUR: 8º -> 6º (subiu 2 posições)
    DEU: 3º -> 2º (subiu 1 posições)
    
     Mudanças do Top 10 entre Durante a Pandemia e Pós-Pandemia:
     Entraram no top 10 em Pós-Pandemia: ESP
     Saíram do top 10 em Pós-Pandemia: HKG
    
     Maiores mudanças de posição:
    VNM: 5º -> 9º (desceu 4 posições)
    MEX: 9º -> 5º (subiu 4 posições)
    NLD: 4º -> 2º (subiu 2 posições)
    DEU: 2º -> 4º (desceu 2 posições)
    IND: 8º -> 7º (subiu 1 posições)
    

# 3. Construção das redes


```python
# função para criar redes por periodo
def create_network_period(df, year, threshold=0):
    
    G = nx.DiGraph(name=f"Rede de Exportação de Máscaras Cirúrgicas - {year}")
    
    # Agrupar total de qty por par de países
    grouped = df.groupby(['Exportador', 'Importador'])['qty'].sum().reset_index()

    for _, row in grouped.iterrows():
        if row['qty'] > threshold:
            G.add_edge(row['Exportador'],
                row['Importador'],
                weight=row['qty']
            )
    
    return G


#função para obter posições geográficas
def get_geo_positions(G):
    pos = {}
    for node in G.nodes():
        if node in coords:
            pos[node] = (coords[node]['Longitude'], coords[node]['Latitude'])
    return pos
```


```python
all_qty = pd.concat([
    periodos["Pré-Pandemia (2015-2018)"]['qty'],
    periodos["Durante a Pandemia (2020-2021)"]['qty'],
    periodos["Pós-Pandemia (2022-2024)"]['qty']
])

print("Resumo estatístico dos pesos (qty):")
print(all_qty.describe(percentiles=[.25, .5, .75, .9, .95, .99]))

```

    Resumo estatístico dos pesos (qty):
    count    3.496220e+05
    mean     3.438018e+04
    std      5.772407e+05
    min      0.000000e+00
    25%      8.900000e+00
    50%      2.000000e+02
    75%      2.870992e+03
    90%      2.148596e+04
    95%      6.436371e+04
    99%      5.055140e+05
    max      7.747843e+07
    Name: qty, dtype: float64
    

#### Testar valores para threshold


```python
for nome, df in periodos.items():
    n_total = df.groupby(['Exportador', 'Importador']).ngroups
    n_filtrado = df[df['qty'] > 60000].groupby(['Exportador', 'Importador']).ngroups
    print(f"{nome}: Total = {n_total}, Após threshold = {n_filtrado}, % mantido = {n_filtrado / n_total:.2%}")

```

    Pré-Pandemia (2015-2018): Total = 8381, Após threshold = 396, % mantido = 4.72%
    Durante a Pandemia (2020-2021): Total = 8067, Após threshold = 609, % mantido = 7.55%
    Pós-Pandemia (2022-2024): Total = 8111, Após threshold = 553, % mantido = 6.82%
    

#### Ler ficheiro com coordenadas e montar redes


```python
# dataFrame de coordenadas dos países
coords_df = pd.read_csv('in-nodes-All.csv')  
coords = coords_df.set_index('Label')[['Longitude', 'Latitude']].to_dict('index')
# montar redes agregadas por período 
networks_periodos = {
    "Pré-Pandemia": create_network_period(periodos["Pré-Pandemia (2015-2018)"], "Pré", threshold=60000),
    "Durante Pandemia": create_network_period(periodos["Durante a Pandemia (2020-2021)"], "Durante", threshold=60000),
    "Pós-Pandemia": create_network_period(periodos["Pós-Pandemia (2022-2024)"], "Pós", threshold=60000),
}
```

## 3.1 Rede Pré-Pandemia - Spring Layout


```python
G = create_network_period(periodos["Pré-Pandemia (2015-2018)"], "Pré", threshold=2000)
#spring layout
pos = nx.spring_layout(G)

plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_color='lightblue',node_size=50,edge_color='gray',alpha=0.6,with_labels=False,arrows=True,arrowsize=10)
plt.title("Rede de Exportações - Pré-Pandemia")
plt.show()
```


    
![png](output_35_0.png)
    


## 3.2 Análise por Períodos - Com coordenadas Geo


```python
# desenhar redes
for title, G in networks_periodos.items():
    plt.figure(figsize=(15, 10))
    pos = get_geo_positions(G)
    
    # desenhar nós 
    nodes = list(pos.keys())
    node_sizes = [G.degree(n)*10 for n in nodes]
    node_color='b'
    edge_width = 0.5
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='w')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color, alpha=0.7)
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.2', edge_color='green', width=edge_width, arrowstyle='->', arrowsize=25)
    
    plt.title(f"Rede de Exportação de Máscaras Cirúrgicas - {title}", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
```


    
![png](output_37_0.png)
    



    
![png](output_37_1.png)
    



    
![png](output_37_2.png)
    



```python
# desenhar redes sem labels e apenas ocm uma amostra das arestas
for title, G in networks_periodos.items():
    pos = get_geo_positions(G)
    
    # filtrar arestas para reduzir densidade, 800 escolhidas de forma aleatória por agora
    all_edges = list(G.edges())
    sample_size = min(800, len(all_edges))  
    edges_sample = random.sample(all_edges, sample_size)
    
    # desenhar nós
    nodes = list(pos.keys())
    node_sizes = [min(G.degree(n)*2, 100) for n in nodes] 
    node_color='b'
    edge_width = 0.5
    
    plt.figure(figsize=(15, 10))
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges_sample, connectionstyle='arc3,rad=0.2', 
                           edge_color='green', width=edge_width, arrowstyle='->', arrowsize=25, alpha=0.6)
    
    plt.title(f"Rede de Exportação de Máscaras Cirúrgicas - {title}", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
```


    
![png](output_38_0.png)
    



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    


#### Inserir Mapa


```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

for title, G in networks_periodos.items():
    pos = get_geo_positions(G)
    
    all_edges = list(G.edges())
    sample_size = min(800, len(all_edges))
    edges_sample = random.sample(all_edges, sample_size)
    
    nodes = list(pos.keys())
    node_sizes = [min(G.degree(n)*2, 100) for n in nodes]
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='b', alpha=0.5,ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_sample, edge_color='darkgreen', width=0.5, alpha=0.4,arrows=True, arrowsize=5, connectionstyle='arc3,rad=0.1', ax=ax)
    
    ax.set_global()
    
    plt.title(f"Rede de Exportação de Máscaras Cirúrgicas - {title}", fontsize=16)
    plt.show()
```


    
![png](output_40_0.png)
    



    
![png](output_40_1.png)
    



    
![png](output_40_2.png)
    


#### filtrar top 10% das arestas com maior peso e eliminar a seleção aleatório das 800 arestas


```python
for title, G in networks_periodos.items():
    pos = get_geo_positions(G)

    edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    n_top = int(len(edges_with_weights) * 0.10)
    edges_sample = [(u, v) for u, v, w in edges_with_weights[:n_top]]

    # espessura proporcional ao peso
    raw_weights = [w for _, _, w in edges_with_weights[:n_top]]
    w_min, w_max = min(raw_weights), max(raw_weights)
    edge_widths = [min(0.5 + w / 100_000_000, 5) for w in raw_weights]

    nodes = list(pos.keys())
    node_sizes = [min(G.degree(n) * 2, 100) for n in nodes]

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='b', alpha=0.5, ax=ax)

    nx.draw_networkx_edges(G,pos,edgelist=edges_sample,width=edge_widths,edge_color='darkgreen',
                           alpha=0.4,arrows=True,arrowsize=5,connectionstyle='arc3,rad=0.1',ax=ax)

    ax.set_global()
    plt.title(f"Rede de Exportação de Máscaras Cirúrgicas - {title}", fontsize=16)
    plt.show()

```


    
![png](output_42_0.png)
    



    
![png](output_42_1.png)
    



    
![png](output_42_2.png)
    


### 3.2.1. Estatísticas básicas


```python
network_stats_per_df = pd.DataFrame([
    {
        'Período': year,
        'Nós': G.number_of_nodes(),
        'Arestas': G.number_of_edges(),
        'Densidade': nx.density(G),
        'Grau Médio': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Componentes Conectadas Fracamente': nx.number_weakly_connected_components(G),
        'Tamanho da Maior Componente': len(max(nx.weakly_connected_components(G), key=len))
    }
    for year, G in networks_periodos.items()
])

# mostrar a tabela
print("\nEstatísticas das Redes de Exportação por Ano:")
display(network_stats_per_df)

```

    
    Estatísticas das Redes de Exportação por Ano:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Período</th>
      <th>Nós</th>
      <th>Arestas</th>
      <th>Densidade</th>
      <th>Grau Médio</th>
      <th>Componentes Conectadas Fracamente</th>
      <th>Tamanho da Maior Componente</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pré-Pandemia</td>
      <td>159</td>
      <td>1136</td>
      <td>0.045219</td>
      <td>14.289308</td>
      <td>1</td>
      <td>159</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Durante Pandemia</td>
      <td>175</td>
      <td>1323</td>
      <td>0.043448</td>
      <td>15.120000</td>
      <td>1</td>
      <td>175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pós-Pandemia</td>
      <td>167</td>
      <td>1361</td>
      <td>0.049095</td>
      <td>16.299401</td>
      <td>1</td>
      <td>167</td>
    </tr>
  </tbody>
</table>
</div>


### 3.2.2. Medidas de Centralidade


```python
# clcular centralidades para cada período e guardar em df
centralidade_periodos = {}

for periodo, G in networks_periodos.items():
    print(f"\n=== Centralidades - {periodo} ===")

    # calcular centralidades
    grau_total = dict(G.degree())
    grau_in = dict(G.in_degree())
    grau_out = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)



    #construir df
    
    df_centralidades = pd.DataFrame({
        'Nó': list(G.nodes()),
        'Grau Total': pd.Series(grau_total),
        'Grau Entrada': pd.Series(grau_in),
        'Grau Saída': pd.Series(grau_out),
        'Betweenness': pd.Series(betweenness),
        'Closeness': pd.Series(closeness),
        'Eigenvector': pd.Series(eigenvector),
        'Page Rank': pd.Series(pagerank),
    })

    df_centralidades = df_centralidades.set_index('Nó')
    df_centralidades = df_centralidades.sort_values('Grau Total', ascending=False)

    #guardar 
    centralidade_periodos[periodo] = df_centralidades

    display(df_centralidades.head(10)) 

```

    
    === Centralidades - Pré-Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grau Total</th>
      <th>Grau Entrada</th>
      <th>Grau Saída</th>
      <th>Betweenness</th>
      <th>Closeness</th>
      <th>Eigenvector</th>
      <th>Page Rank</th>
    </tr>
    <tr>
      <th>Nó</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHN</th>
      <td>152</td>
      <td>19</td>
      <td>133</td>
      <td>0.114124</td>
      <td>0.221711</td>
      <td>0.154300</td>
      <td>0.012788</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>107</td>
      <td>37</td>
      <td>70</td>
      <td>0.076492</td>
      <td>0.261302</td>
      <td>0.260183</td>
      <td>0.073282</td>
    </tr>
    <tr>
      <th>HKG</th>
      <td>102</td>
      <td>10</td>
      <td>92</td>
      <td>0.023828</td>
      <td>0.201833</td>
      <td>0.083668</td>
      <td>0.004925</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>88</td>
      <td>33</td>
      <td>55</td>
      <td>0.098182</td>
      <td>0.258990</td>
      <td>0.215394</td>
      <td>0.044367</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>78</td>
      <td>9</td>
      <td>69</td>
      <td>0.035096</td>
      <td>0.199087</td>
      <td>0.070155</td>
      <td>0.003762</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>77</td>
      <td>14</td>
      <td>63</td>
      <td>0.022569</td>
      <td>0.207559</td>
      <td>0.131956</td>
      <td>0.005606</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>69</td>
      <td>31</td>
      <td>38</td>
      <td>0.034094</td>
      <td>0.243882</td>
      <td>0.220822</td>
      <td>0.035796</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>67</td>
      <td>24</td>
      <td>43</td>
      <td>0.019269</td>
      <td>0.230440</td>
      <td>0.188709</td>
      <td>0.023910</td>
    </tr>
    <tr>
      <th>POL</th>
      <td>56</td>
      <td>23</td>
      <td>33</td>
      <td>0.012622</td>
      <td>0.232268</td>
      <td>0.197858</td>
      <td>0.026402</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>56</td>
      <td>25</td>
      <td>31</td>
      <td>0.019892</td>
      <td>0.237934</td>
      <td>0.190068</td>
      <td>0.023499</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Centralidades - Durante Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grau Total</th>
      <th>Grau Entrada</th>
      <th>Grau Saída</th>
      <th>Betweenness</th>
      <th>Closeness</th>
      <th>Eigenvector</th>
      <th>Page Rank</th>
    </tr>
    <tr>
      <th>Nó</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHN</th>
      <td>197</td>
      <td>34</td>
      <td>163</td>
      <td>0.198836</td>
      <td>0.284665</td>
      <td>0.188756</td>
      <td>0.017235</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>114</td>
      <td>53</td>
      <td>61</td>
      <td>0.064766</td>
      <td>0.335213</td>
      <td>0.277015</td>
      <td>0.065098</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>109</td>
      <td>45</td>
      <td>64</td>
      <td>0.084185</td>
      <td>0.306563</td>
      <td>0.220061</td>
      <td>0.094732</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>87</td>
      <td>15</td>
      <td>72</td>
      <td>0.021002</td>
      <td>0.235972</td>
      <td>0.112164</td>
      <td>0.005353</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>87</td>
      <td>41</td>
      <td>46</td>
      <td>0.031644</td>
      <td>0.296428</td>
      <td>0.233230</td>
      <td>0.052897</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>82</td>
      <td>35</td>
      <td>47</td>
      <td>0.017209</td>
      <td>0.273800</td>
      <td>0.206814</td>
      <td>0.021847</td>
    </tr>
    <tr>
      <th>HKG</th>
      <td>76</td>
      <td>24</td>
      <td>52</td>
      <td>0.017393</td>
      <td>0.256199</td>
      <td>0.128535</td>
      <td>0.009771</td>
    </tr>
    <tr>
      <th>GBR</th>
      <td>68</td>
      <td>33</td>
      <td>35</td>
      <td>0.010632</td>
      <td>0.269683</td>
      <td>0.197584</td>
      <td>0.015855</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>66</td>
      <td>11</td>
      <td>55</td>
      <td>0.009343</td>
      <td>0.228457</td>
      <td>0.065748</td>
      <td>0.004746</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>64</td>
      <td>30</td>
      <td>34</td>
      <td>0.014006</td>
      <td>0.265688</td>
      <td>0.202409</td>
      <td>0.027568</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Centralidades - Pós-Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grau Total</th>
      <th>Grau Entrada</th>
      <th>Grau Saída</th>
      <th>Betweenness</th>
      <th>Closeness</th>
      <th>Eigenvector</th>
      <th>Page Rank</th>
    </tr>
    <tr>
      <th>Nó</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHN</th>
      <td>176</td>
      <td>29</td>
      <td>147</td>
      <td>0.139261</td>
      <td>0.258890</td>
      <td>0.170584</td>
      <td>0.010085</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>111</td>
      <td>47</td>
      <td>64</td>
      <td>0.064736</td>
      <td>0.305737</td>
      <td>0.251804</td>
      <td>0.064468</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>107</td>
      <td>43</td>
      <td>64</td>
      <td>0.106485</td>
      <td>0.305737</td>
      <td>0.206620</td>
      <td>0.091629</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>95</td>
      <td>19</td>
      <td>76</td>
      <td>0.036093</td>
      <td>0.241372</td>
      <td>0.129955</td>
      <td>0.005128</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>82</td>
      <td>38</td>
      <td>44</td>
      <td>0.028375</td>
      <td>0.284092</td>
      <td>0.215068</td>
      <td>0.045997</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>77</td>
      <td>30</td>
      <td>47</td>
      <td>0.010932</td>
      <td>0.260995</td>
      <td>0.180750</td>
      <td>0.019284</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>72</td>
      <td>12</td>
      <td>60</td>
      <td>0.016060</td>
      <td>0.222933</td>
      <td>0.071619</td>
      <td>0.002990</td>
    </tr>
    <tr>
      <th>GBR</th>
      <td>72</td>
      <td>33</td>
      <td>39</td>
      <td>0.016308</td>
      <td>0.267520</td>
      <td>0.199229</td>
      <td>0.015623</td>
    </tr>
    <tr>
      <th>POL</th>
      <td>72</td>
      <td>34</td>
      <td>38</td>
      <td>0.011436</td>
      <td>0.269768</td>
      <td>0.204632</td>
      <td>0.028959</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>67</td>
      <td>32</td>
      <td>35</td>
      <td>0.011521</td>
      <td>0.265309</td>
      <td>0.205465</td>
      <td>0.025458</td>
    </tr>
  </tbody>
</table>
</div>



```python
for periodo, G in networks_periodos.items():
    print(f"\n=== Centralidades - {periodo} ===")
    
    # maior componente fortemente conectada
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc).copy()
    
    # excentricidade para a maior componente fortemente conectada
    excentricidade = nx.eccentricity(G_scc)
    
    #df com os resultados
    df_excentricidade = pd.DataFrame({
        'Excentricidade': pd.Series(excentricidade),
    })
    df_excentricidade = df_excentricidade.sort_values('Excentricidade', ascending=True)
    display(df_excentricidade.head(10))
    raio = min(excentricidade.values())
    print(f"Raio do grafo para {periodo}: {raio}")
```

    
    === Centralidades - Pré-Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GBR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>2</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>2</td>
    </tr>
    <tr>
      <th>POL</th>
      <td>2</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>VNM</th>
      <td>2</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Raio do grafo para Pré-Pandemia: 2
    
    === Centralidades - Durante Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>UKR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>2</td>
    </tr>
    <tr>
      <th>SGP</th>
      <td>2</td>
    </tr>
    <tr>
      <th>GBR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>IDN</th>
      <td>2</td>
    </tr>
    <tr>
      <th>SVK</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Raio do grafo para Durante Pandemia: 1
    
    === Centralidades - Pós-Pandemia ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GBR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>TUR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CHE</th>
      <td>2</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>2</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NLD</th>
      <td>2</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Raio do grafo para Pós-Pandemia: 2
    

### 3.2.3 - Propriedades de Conectividade


```python
for periodo, G in networks_periodos.items():
    print(f'\n\n--- Período {periodo} ---')

    print('É fortemente conexo?', nx.is_strongly_connected(G))
    print('É fracamente conexo?', nx.is_weakly_connected(G))
    
    print('Número de componentes fortemente conexas:', nx.number_strongly_connected_components(G))
    print('Número de componentes fracamente conexas:', nx.number_weakly_connected_components(G))

    print('\nTop 3 componentes fortemente conexas:')
    for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)[:3]:
        print(c)

    print('\nTop 3 componentes fracamente conexas:')
    for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)[:3]:
        print(c)

```

    
    
    --- Período Pré-Pandemia ---
    É fortemente conexo? False
    É fracamente conexo? True
    Número de componentes fortemente conexas: 91
    Número de componentes fracamente conexas: 1
    
    Top 3 componentes fortemente conexas:
    {'GBR', 'ISR', 'MYS', 'MEX', 'PRY', 'BEL', 'LTU', 'DOM', 'HUN', 'SLV', 'PHL', 'ARE', 'KOR', 'MAR', 'ROU', 'SVK', 'PAK', 'NLD', 'DEU', 'IRL', 'UKR', 'IDN', 'MMR', 'EST', 'SVN', 'COL', 'CHN', 'MDG', 'IND', 'SGP', 'CHL', 'CHE', 'SAU', 'TUR', 'PRT', 'MDA', 'BLR', 'LUX', 'LAO', 'KAZ', 'BRA', 'ESP', 'BHR', 'HRV', 'NOR', 'BGR', 'LVA', 'MKD', 'THA', 'USA', 'CZE', 'DNK', 'POL', 'SWE', 'SRB', 'AUS', 'ITA', 'KWT', 'ZAF', 'JPN', 'KHM', 'ECU', 'VNM', 'RUS', 'HKG', 'FRA', 'PER', 'FIN', 'BIH'}
    {'JOR'}
    {'QAT'}
    
    Top 3 componentes fracamente conexas:
    {'UGA', 'MEX', 'MYS', 'URY', 'BEL', 'SLV', 'PHL', 'ARE', 'AZE', 'DEU', 'IRL', 'COD', 'MMR', 'AUT', 'GMB', 'TGO', 'MDG', 'SDN', 'SAU', 'MUS', 'TUR', 'KAZ', 'LUX', 'BWA', 'BRA', 'NOR', 'MRT', 'SRB', 'SYR', 'JPN', 'KHM', 'BHS', 'RUS', 'DJI', 'UZB', 'HKG', 'TTO', 'GUM', 'AFG', 'RWA', 'CHN', 'IND', 'SGP', 'MDV', 'PRT', 'BHR', 'SEN', 'SOM', 'ESP', 'GTM', 'SWZ', 'IRN', 'USA', 'COM', 'DNK', 'CRI', 'EGY', 'ITA', 'NAM', 'YEM', 'GHA', 'CYP', 'LBN', 'MAC', 'GUY', 'ALB', 'ETH', 'GAB', 'KEN', 'NZL', 'TZA', 'ZMB', 'DOM', 'PRY', 'MNG', 'CUB', 'ROU', 'SVK', 'NLD', 'UKR', 'DZA', 'EST', 'COL', 'FJI', 'CHE', 'CYM', 'BOL', 'PYF', 'BRN', 'HRV', 'THA', 'JAM', 'POL', 'ARG', 'JOR', 'MLT', 'CAN', 'KGZ', 'PAN', 'AND', 'FIN', 'GBR', 'NIC', 'ISR', 'COG', 'VEN', 'TKM', 'BRB', 'NGA', 'LTU', 'HUN', 'TJK', 'KOR', 'MAR', 'IRQ', 'PAK', 'GRC', 'LSO', 'ISL', 'IDN', 'SVN', 'NPL', 'PRK', 'CHL', 'HND', 'BGD', 'LBY', 'PNG', 'TUN', 'LAO', 'MDA', 'BLR', 'CIV', 'QAT', 'BEN', 'MOZ', 'GEO', 'MWI', 'BGR', 'LVA', 'MKD', 'CZE', 'SWE', 'KWT', 'CMR', 'HTI', 'AUS', 'ZAF', 'ECU', 'VNM', 'NCL', 'FRA', 'OMN', 'LKA', 'PER', 'AGO', 'ZWE', 'GIN', 'BIH'}
    
    
    --- Período Durante Pandemia ---
    É fortemente conexo? False
    É fracamente conexo? True
    Número de componentes fortemente conexas: 96
    Número de componentes fracamente conexas: 1
    
    Top 3 componentes fortemente conexas:
    {'URY', 'MEX', 'MYS', 'BEL', 'SLV', 'PHL', 'DEU', 'IRL', 'MMR', 'MDG', 'TUR', 'LUX', 'BRA', 'NOR', 'SRB', 'JPN', 'KHM', 'RUS', 'UZB', 'HKG', 'CHN', 'IND', 'SGP', 'PRT', 'BHR', 'ESP', 'GTM', 'USA', 'DNK', 'EGY', 'ITA', 'GHA', 'CYP', 'MAC', 'PRY', 'DOM', 'ROU', 'SVK', 'NLD', 'UKR', 'EST', 'COL', 'CHE', 'BOL', 'HRV', 'THA', 'POL', 'ARG', 'CAN', 'FIN', 'GBR', 'NIC', 'ISR', 'LTU', 'HUN', 'KOR', 'MAR', 'PAK', 'GRC', 'IDN', 'SVN', 'CHL', 'TUN', 'LAO', 'MDA', 'BLR', 'GEO', 'ARM', 'BGR', 'MKD', 'LVA', 'CZE', 'SWE', 'AUS', 'ZAF', 'ECU', 'VNM', 'FRA', 'PER', 'BIH'}
    {'AFG'}
    {'AGO'}
    
    Top 3 componentes fracamente conexas:
    {'UGA', 'URY', 'MYS', 'MEX', 'BEL', 'SLV', 'PHL', 'ARE', 'AZE', 'DEU', 'IRL', 'COD', 'MMR', 'AUT', 'GMB', 'TGO', 'MDG', 'SDN', 'MUS', 'SAU', 'TUR', 'KAZ', 'LUX', 'BWA', 'BRA', 'MRT', 'NOR', 'LBR', 'TLS', 'SRB', 'SYR', 'JPN', 'KHM', 'BHS', 'CPV', 'RUS', 'DJI', 'UZB', 'HKG', 'TTO', 'AFG', 'MNE', 'RWA', 'CHN', 'SGP', 'IND', 'TCD', 'MDV', 'PRT', 'BHR', 'SEN', 'SUR', 'GRD', 'SOM', 'ESP', 'GTM', 'SWZ', 'IRN', 'USA', 'COM', 'DNK', 'CRI', 'EGY', 'ITA', 'NAM', 'YEM', 'BLZ', 'GHA', 'BFA', 'CYP', 'GUY', 'LBN', 'MAC', 'ALB', 'SLE', 'ETH', 'GAB', 'KEN', 'NZL', 'PRY', 'DOM', 'TZA', 'ZMB', 'MNG', 'CUB', 'ROU', 'SVK', 'NLD', 'UKR', 'DZA', 'EST', 'COL', 'NER', 'FJI', 'CHE', 'CYM', 'BOL', 'PYF', 'BRN', 'HRV', 'THA', 'JAM', 'POL', 'ARG', 'JOR', 'MLT', 'CAN', 'DMA', 'KGZ', 'PAN', 'AND', 'FIN', 'GBR', 'NIC', 'COG', 'ISR', 'TKM', 'VEN', 'BRB', 'NGA', 'LTU', 'HUN', 'TJK', 'KOR', 'MAR', 'PSE', 'PAK', 'GRC', 'IRQ', 'LSO', 'ISL', 'IDN', 'SVN', 'NPL', 'PRK', 'CHL', 'HND', 'BGD', 'LBY', 'TUN', 'PNG', 'LAO', 'MDA', 'BLR', 'CIV', 'BEN', 'MWI', 'MOZ', 'GEO', 'QAT', 'ARM', 'SSD', 'BGR', 'LVA', 'MKD', 'VCT', 'CZE', 'SWE', 'CMR', 'HTI', 'KWT', 'AUS', 'MLI', 'ZAF', 'ECU', 'VNM', 'NCL', 'FRA', 'PER', 'LKA', 'OMN', 'AGO', 'ZWE', 'GIN', 'BIH'}
    
    
    --- Período Pós-Pandemia ---
    É fortemente conexo? False
    É fracamente conexo? True
    Número de componentes fortemente conexas: 94
    Número de componentes fracamente conexas: 1
    
    Top 3 componentes fortemente conexas:
    {'GBR', 'ISR', 'NZL', 'MEX', 'URY', 'PRY', 'BEL', 'LTU', 'DOM', 'HUN', 'SLV', 'MYS', 'PHL', 'KOR', 'MAR', 'ROU', 'SVK', 'PAK', 'GRC', 'DEU', 'IRL', 'NLD', 'UKR', 'IDN', 'AUT', 'MMR', 'EST', 'SVN', 'COL', 'CHN', 'MDG', 'IND', 'SGP', 'CHL', 'CHE', 'SAU', 'TUN', 'BOL', 'TUR', 'LAO', 'PRT', 'KAZ', 'MDA', 'BHR', 'BRA', 'ESP', 'GEO', 'GTM', 'HRV', 'NOR', 'BGR', 'LVA', 'MKD', 'THA', 'USA', 'CZE', 'DNK', 'POL', 'ARG', 'SWE', 'AUS', 'ITA', 'SRB', 'ZAF', 'JPN', 'KHM', 'VNM', 'CAN', 'UZB', 'HKG', 'FRA', 'GHA', 'FIN', 'BIH'}
    {'ECU'}
    {'ARE'}
    
    Top 3 componentes fracamente conexas:
    {'UGA', 'URY', 'MEX', 'MYS', 'BEL', 'SLV', 'PHL', 'ARE', 'AZE', 'DEU', 'IRL', 'COD', 'MMR', 'AUT', 'TGO', 'MDG', 'SDN', 'MUS', 'SAU', 'TUR', 'KAZ', 'LUX', 'BWA', 'BRA', 'MRT', 'NOR', 'LBR', 'SRB', 'SYR', 'JPN', 'KHM', 'BHS', 'RUS', 'DJI', 'UZB', 'HKG', 'TTO', 'AFG', 'MNE', 'GRL', 'FRO', 'RWA', 'CHN', 'IND', 'SGP', 'MDV', 'PRT', 'BHR', 'SEN', 'SUR', 'SOM', 'ESP', 'GTM', 'SWZ', 'IRN', 'USA', 'DNK', 'CRI', 'EGY', 'ITA', 'NAM', 'BLZ', 'YEM', 'GHA', 'BFA', 'CYP', 'GUY', 'LBN', 'MAC', 'ALB', 'ETH', 'GAB', 'KEN', 'NZL', 'PRY', 'DOM', 'TZA', 'ZMB', 'MNG', 'CUB', 'ROU', 'SVK', 'NLD', 'UKR', 'DZA', 'EST', 'COL', 'FJI', 'CHE', 'CYM', 'BOL', 'PYF', 'BRN', 'HRV', 'THA', 'JAM', 'POL', 'ARG', 'JOR', 'MLT', 'CAN', 'KGZ', 'PAN', 'AND', 'GNQ', 'FIN', 'GBR', 'NIC', 'ISR', 'COG', 'TKM', 'VEN', 'BRB', 'NGA', 'LTU', 'HUN', 'TJK', 'KOR', 'MAR', 'PAK', 'GRC', 'IRQ', 'LSO', 'ISL', 'IDN', 'SVN', 'NPL', 'PRK', 'CHL', 'HND', 'BGD', 'LBY', 'PNG', 'TUN', 'LAO', 'MDA', 'BLR', 'CIV', 'BEN', 'MWI', 'MOZ', 'GEO', 'QAT', 'ARM', 'BGR', 'LVA', 'MKD', 'CZE', 'SWE', 'CMR', 'HTI', 'KWT', 'AUS', 'MLI', 'CUW', 'ZAF', 'ECU', 'VNM', 'NCL', 'FRA', 'LKA', 'OMN', 'PER', 'AGO', 'ZWE', 'GIN', 'BIH'}
    

### 3.2.4 - Medidas de Distância


```python
G_pre = create_network_period(periodos["Pré-Pandemia (2015-2018)"], "Pré", threshold=0)
G_durante = create_network_period(periodos["Durante a Pandemia (2020-2021)"], "Durante", threshold=0)
G_pos = create_network_period(periodos["Pós-Pandemia (2022-2024)"], "Pós", threshold=0)

# dicionários de graus
grau_entrada_pre = dict(G_pre.in_degree())
grau_saida_pre = dict(G_pre.out_degree())
grau_entrada_durante = dict(G_durante.in_degree())
grau_saida_durante = dict(G_durante.out_degree())

# top países por grau de entrada
top_entrada_pre = pd.Series(grau_entrada_pre).sort_values(ascending=False).head(10)
top_entrada_durante = pd.Series(grau_entrada_durante).sort_values(ascending=False).head(10)

# top países por grau de entrada
top_saida_pre = pd.Series(grau_saida_pre).sort_values(ascending=False).head(10)
top_saida_durante = pd.Series(grau_saida_durante).sort_values(ascending=False).head(10)

# plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Top 10 Países por Grau de Entrada e Saída", fontsize=16)

top_entrada_pre.plot(kind="bar", ax=axes[0, 0], title="Entrada - Pré-pandemia", color='steelblue')
top_entrada_durante.plot(kind="bar", ax=axes[0, 1], title="Entrada - Durante pandemia", color='darkorange')
top_saida_pre.plot(kind="bar", ax=axes[1, 0], title="Saída - Pré-pandemia", color='steelblue')
top_saida_durante.plot(kind="bar", ax=axes[1, 1], title="Saída - Durante pandemia", color='darkorange')

for ax in axes.flatten():
    ax.set_ylabel("Número de conexões")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```


    
![png](output_51_0.png)
    



```python
def calcular_medidas_distancia(G, periodo, top_n=10):
    print(f"\n=== Medidas de Distância - {periodo} ===")
    
    # identificar a maior componente fortemente conectada (SCC)
    scc = list(nx.strongly_connected_components(G))
    scc.sort(key=len, reverse=True)  # ordenar por tamanho
    maior_scc = scc[0]
    G_scc = G.subgraph(maior_scc).copy()  # fazer o subgrafo da primeira - maior
    
    print(f"Maior componente fortemente conectada tem {len(G_scc)} nós.")
    print()

    #grande parte do código destas medidas -  abaixo e algum acima - é das aulas, dos notebooks do professor, adaptei apenas
    # Excentricidade: Top 5 menor e maior
    excent = nx.eccentricity(G_scc)
    excent_df = pd.DataFrame(list(excent.items()), columns=['Nó', 'Excentricidade'])
    excent_df = excent_df.sort_values('Excentricidade')
    
    print("Top 5 nós com menor excentricidade (mais centrais):")
    display(excent_df.head(5))
    print("Top 5 nós com maior excentricidade (mais periféricos):")
    display(excent_df.tail(5))
    print()
    
    # Raio
    r = nx.radius(G_scc)
    print('Raio de G:', r)
    print()
    
    # Diâmetro
    d = nx.diameter(G_scc)
    print('Diâmetro de G:', d)
    print()
    
    # Centro
    centro = nx.center(G_scc)
    print('Centro de G (nodos com excentricidade = raio):', centro)
    print()
    
    # Periferia
    periferia = nx.periphery(G_scc)
    print('Periferia de G (nodos com excentricidade = diâmetro):', periferia)
    print()
    
    # Graus de entrada e saída: Top 10
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    in_deg_df = pd.DataFrame(list(in_deg.items()), columns=['Nó', 'Grau de Entrada'])
    out_deg_df = pd.DataFrame(list(out_deg.items()), columns=['Nó', 'Grau de Saída'])
    
    in_deg_df = in_deg_df.sort_values('Grau de Entrada', ascending=False)
    out_deg_df = out_deg_df.sort_values('Grau de Saída', ascending=False)
    
    print(f"Top {top_n} nós com maior grau de entrada:")
    display(in_deg_df.head(top_n))
    print(f"Top {top_n} nós com maior grau de saída:")
    display(out_deg_df.head(top_n))
    print()
    
    # Grau médio
    gm_in = sum(in_deg.values()) / len(G.nodes)
    gm_out = sum(out_deg.values()) / len(G.nodes)
    print('Grau médio de entrada de G:', gm_in)
    print('Grau médio de saída de G:', gm_out)
    print()
    
    # Average path length
    apl = nx.average_shortest_path_length(G_scc)
    print('Average path length de G:', apl)
    print()

# aplicar a cada período
for periodo, G in networks_periodos.items():
    calcular_medidas_distancia(G, periodo, top_n=10)
```

    
    === Medidas de Distância - Pré-Pandemia ===
    Maior componente fortemente conectada tem 69 nós.
    
    Top 5 nós com menor excentricidade (mais centrais):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GBR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CHN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DEU</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NLD</td>
      <td>2</td>
    </tr>
    <tr>
      <th>52</th>
      <td>POL</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Top 5 nós com maior excentricidade (mais periféricos):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57</th>
      <td>KWT</td>
      <td>5</td>
    </tr>
    <tr>
      <th>61</th>
      <td>ECU</td>
      <td>5</td>
    </tr>
    <tr>
      <th>66</th>
      <td>PER</td>
      <td>5</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SAU</td>
      <td>6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>BHR</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


    
    Raio de G: 2
    
    Diâmetro de G: 7
    
    Centro de G (nodos com excentricidade = raio): ['GBR', 'NLD', 'DEU', 'CHN', 'USA', 'POL', 'SWE', 'ITA', 'VNM', 'HKG', 'FRA']
    
    Periferia de G (nodos com excentricidade = diâmetro): ['BHR']
    
    Top 10 nós com maior grau de entrada:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Entrada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>DEU</td>
      <td>37</td>
    </tr>
    <tr>
      <th>21</th>
      <td>USA</td>
      <td>33</td>
    </tr>
    <tr>
      <th>29</th>
      <td>FRA</td>
      <td>31</td>
    </tr>
    <tr>
      <th>30</th>
      <td>GBR</td>
      <td>26</td>
    </tr>
    <tr>
      <th>32</th>
      <td>ITA</td>
      <td>25</td>
    </tr>
    <tr>
      <th>44</th>
      <td>RUS</td>
      <td>25</td>
    </tr>
    <tr>
      <th>34</th>
      <td>NLD</td>
      <td>24</td>
    </tr>
    <tr>
      <th>40</th>
      <td>SWE</td>
      <td>23</td>
    </tr>
    <tr>
      <th>35</th>
      <td>POL</td>
      <td>23</td>
    </tr>
    <tr>
      <th>26</th>
      <td>DNK</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>


    Top 10 nós com maior grau de saída:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Saída</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>CHN</td>
      <td>133</td>
    </tr>
    <tr>
      <th>84</th>
      <td>HKG</td>
      <td>92</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DEU</td>
      <td>70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IND</td>
      <td>69</td>
    </tr>
    <tr>
      <th>130</th>
      <td>TUR</td>
      <td>63</td>
    </tr>
    <tr>
      <th>21</th>
      <td>USA</td>
      <td>55</td>
    </tr>
    <tr>
      <th>34</th>
      <td>NLD</td>
      <td>43</td>
    </tr>
    <tr>
      <th>29</th>
      <td>FRA</td>
      <td>38</td>
    </tr>
    <tr>
      <th>135</th>
      <td>VNM</td>
      <td>33</td>
    </tr>
    <tr>
      <th>35</th>
      <td>POL</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>


    
    Grau médio de entrada de G: 7.144654088050315
    Grau médio de saída de G: 7.144654088050315
    
    Average path length de G: 2.2440323955669226
    
    
    === Medidas de Distância - Durante Pandemia ===
    Maior componente fortemente conectada tem 80 nós.
    
    Top 5 nós com menor excentricidade (mais centrais):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>CHN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>UKR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>IND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SGP</td>
      <td>2</td>
    </tr>
    <tr>
      <th>50</th>
      <td>GBR</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Top 5 nós com maior excentricidade (mais periféricos):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>HRV</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>DNK</td>
      <td>3</td>
    </tr>
    <tr>
      <th>79</th>
      <td>BIH</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CYP</td>
      <td>4</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ARG</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    Raio de G: 1
    
    Diâmetro de G: 4
    
    Centro de G (nodos com excentricidade = raio): ['CHN']
    
    Periferia de G (nodos com excentricidade = diâmetro): ['CYP', 'ARG']
    
    Top 10 nós com maior grau de entrada:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Entrada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>DEU</td>
      <td>53</td>
    </tr>
    <tr>
      <th>12</th>
      <td>USA</td>
      <td>45</td>
    </tr>
    <tr>
      <th>22</th>
      <td>FRA</td>
      <td>41</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NLD</td>
      <td>35</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CHN</td>
      <td>34</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GBR</td>
      <td>33</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ITA</td>
      <td>30</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BEL</td>
      <td>28</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ESP</td>
      <td>27</td>
    </tr>
    <tr>
      <th>30</th>
      <td>POL</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>


    Top 10 nós com maior grau de saída:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Saída</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>CHN</td>
      <td>163</td>
    </tr>
    <tr>
      <th>43</th>
      <td>TUR</td>
      <td>72</td>
    </tr>
    <tr>
      <th>12</th>
      <td>USA</td>
      <td>64</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DEU</td>
      <td>61</td>
    </tr>
    <tr>
      <th>8</th>
      <td>IND</td>
      <td>55</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HKG</td>
      <td>52</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NLD</td>
      <td>47</td>
    </tr>
    <tr>
      <th>22</th>
      <td>FRA</td>
      <td>46</td>
    </tr>
    <tr>
      <th>160</th>
      <td>VNM</td>
      <td>40</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GBR</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>


    
    Grau médio de entrada de G: 7.56
    Grau médio de saída de G: 7.56
    
    Average path length de G: 2.0765822784810126
    
    
    === Medidas de Distância - Pós-Pandemia ===
    Maior componente fortemente conectada tem 74 nós.
    
    Top 5 nós com menor excentricidade (mais centrais):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GBR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>TUR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ESP</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CHE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>IND</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    Top 5 nós com maior excentricidade (mais periféricos):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Excentricidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>URY</td>
      <td>4</td>
    </tr>
    <tr>
      <th>68</th>
      <td>UZB</td>
      <td>4</td>
    </tr>
    <tr>
      <th>35</th>
      <td>SAU</td>
      <td>4</td>
    </tr>
    <tr>
      <th>58</th>
      <td>ARG</td>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BOL</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    
    Raio de G: 2
    
    Diâmetro de G: 5
    
    Centro de G (nodos com excentricidade = raio): ['GBR', 'HUN', 'MYS', 'KOR', 'SVK', 'PAK', 'DEU', 'NLD', 'CHN', 'IND', 'CHE', 'TUR', 'ESP', 'USA', 'CZE', 'DNK', 'POL', 'SWE', 'ITA', 'KHM', 'VNM', 'CAN', 'HKG', 'FRA']
    
    Periferia de G (nodos com excentricidade = diâmetro): ['BOL']
    
    Top 10 nós com maior grau de entrada:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Entrada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>DEU</td>
      <td>47</td>
    </tr>
    <tr>
      <th>15</th>
      <td>USA</td>
      <td>43</td>
    </tr>
    <tr>
      <th>21</th>
      <td>FRA</td>
      <td>38</td>
    </tr>
    <tr>
      <th>26</th>
      <td>POL</td>
      <td>34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GBR</td>
      <td>33</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ITA</td>
      <td>32</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NLD</td>
      <td>30</td>
    </tr>
    <tr>
      <th>30</th>
      <td>BEL</td>
      <td>29</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CHN</td>
      <td>29</td>
    </tr>
    <tr>
      <th>40</th>
      <td>SWE</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>


    Top 10 nós com maior grau de saída:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Grau de Saída</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CHN</td>
      <td>147</td>
    </tr>
    <tr>
      <th>55</th>
      <td>TUR</td>
      <td>76</td>
    </tr>
    <tr>
      <th>20</th>
      <td>DEU</td>
      <td>64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>USA</td>
      <td>64</td>
    </tr>
    <tr>
      <th>88</th>
      <td>IND</td>
      <td>60</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NLD</td>
      <td>47</td>
    </tr>
    <tr>
      <th>21</th>
      <td>FRA</td>
      <td>44</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GBR</td>
      <td>39</td>
    </tr>
    <tr>
      <th>26</th>
      <td>POL</td>
      <td>38</td>
    </tr>
    <tr>
      <th>31</th>
      <td>DNK</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>


    
    Grau médio de entrada de G: 8.149700598802395
    Grau médio de saída de G: 8.149700598802395
    
    Average path length de G: 2.0510921880784894
    
    

### 3.2.5 - Comunidades


```python
# dicionários para guardar resultados por período
communities_by_period = {}
colors_by_period = {}
modularity_by_period = {}
partition_quality_by_period = {}

for period, G in networks_periodos.items():
    print(f"\nPeriodo: {period}")
    
    # aplicar Louvain diretamente no grafo direcionado com pesos
    com_louvain = list(nx.community.louvain_communities(G, weight='weight', seed=42))
    communities_by_period[period] = com_louvain

    print(f"Número de comunidades: {len(com_louvain)}")
    for i, com in enumerate(com_louvain):
        print(f" - Comunidade {i+1}: {len(com)} nós")

    # cor aos nós
    node_color = [(0.5, 0.5, 0.5) for _ in G.nodes()]
    for i, v in enumerate(G.nodes()):
        for j in range(len(com_louvain)):
            if v in com_louvain[j]:
                node_color[i] = (0.25*j/10, 2.1*j/10, 1-2*j/10) #-> ideia encontrada no stackoverflow
    colors_by_period[period] = node_color

    #modularidade e qualidade
    m1 = nx.community.modularity(G, communities=com_louvain, weight='weight', resolution=1)
    q1 = nx.community.partition_quality(G, partition=com_louvain)

    modularity_by_period[period] = m1
    partition_quality_by_period[period] = q1

    print(f"Índice de modularidade: {m1:.3f}")
    print(f"Cobertura: {q1[0]:.3f} | Performance: {q1[1]:.3f}")

    # desenhar grafo com comunidades
    pos = get_geo_positions(G)
    plt.figure(figsize=(20,15))
    nx.draw_networkx(G, pos, node_color=node_color, with_labels=True, node_size=600,
                     node_shape='o', alpha=0.8, edge_color='gray', width=0.5)
    plt.title(f"Comunidades Louvain - {period}")
    plt.show()

    # bgrafos por comunidade
    f, axs = plt.subplots(nrows=math.ceil(len(com_louvain)/2), ncols=2, figsize=(22, 20))
    axs = axs.flatten()
    for i in range(len(com_louvain)):
        G1 = G.subgraph(com_louvain[i])
        node_color1 = [node_color[j] for j, v in enumerate(G.nodes()) if v in com_louvain[i]]
        nx.draw_networkx_nodes(G1, pos, nodelist=com_louvain[i], node_color=node_color1,node_size=600, node_shape='o', alpha=0.8, ax=axs[i])
        nx.draw_networkx(G1, pos, with_labels=True, node_size=0, edge_color='gray',width=0.5, ax=axs[i])
        axs[i].set_title(f"Comunidade {i+1} - {period}")
    plt.tight_layout()
    plt.show()

```

    
    Periodo: Pré-Pandemia
    Número de comunidades: 4
     - Comunidade 1: 57 nós
     - Comunidade 2: 62 nós
     - Comunidade 3: 38 nós
     - Comunidade 4: 2 nós
    Índice de modularidade: 0.232
    Cobertura: 0.644 | Performance: 0.679
    


    
![png](output_54_1.png)
    



    
![png](output_54_2.png)
    


    
    Periodo: Durante Pandemia
    Número de comunidades: 4
     - Comunidade 1: 12 nós
     - Comunidade 2: 28 nós
     - Comunidade 3: 79 nós
     - Comunidade 4: 56 nós
    Índice de modularidade: 0.193
    Cobertura: 0.675 | Performance: 0.683
    


    
![png](output_54_4.png)
    



    
![png](output_54_5.png)
    


    
    Periodo: Pós-Pandemia
    Número de comunidades: 4
     - Comunidade 1: 19 nós
     - Comunidade 2: 89 nós
     - Comunidade 3: 56 nós
     - Comunidade 4: 3 nós
    Índice de modularidade: 0.306
    Cobertura: 0.686 | Performance: 0.612
    


    
![png](output_54_7.png)
    



    
![png](output_54_8.png)
    



```python
networks = {'Pré': G_pre, 'Durante': G_durante, 'Pós': G_pos}
summary_data = []

for year, G in networks.items():
    stats = {
        'Período': year,
        'Nº Nós': G.number_of_nodes(),
        'Nº Arestas': G.number_of_edges(),
        'Densidade': nx.density(G),
        'Grau Médio': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Grau Máximo': max(dict(G.degree()).values()),
        'Componentes Fortes': nx.number_strongly_connected_components(G) if G.is_directed() else '-',
        'Clustering Médio': nx.average_clustering(G.to_undirected())
    }
    summary_data.append(stats)

df_summary = pd.DataFrame(summary_data)
display(df_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Período</th>
      <th>Nº Nós</th>
      <th>Nº Arestas</th>
      <th>Densidade</th>
      <th>Grau Médio</th>
      <th>Grau Máximo</th>
      <th>Componentes Fortes</th>
      <th>Clustering Médio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pré</td>
      <td>230</td>
      <td>7584</td>
      <td>0.143991</td>
      <td>65.947826</td>
      <td>291</td>
      <td>101</td>
      <td>0.795445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Durante</td>
      <td>228</td>
      <td>8057</td>
      <td>0.155673</td>
      <td>70.675439</td>
      <td>311</td>
      <td>102</td>
      <td>0.803318</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pós</td>
      <td>230</td>
      <td>8106</td>
      <td>0.153902</td>
      <td>70.486957</td>
      <td>290</td>
      <td>109</td>
      <td>0.817830</td>
    </tr>
  </tbody>
</table>
</div>



```python
clustering_pre = nx.transitivity(G_pre.to_undirected())
clustering_dur = nx.transitivity(G_durante.to_undirected())
clustering_pos = nx.transitivity(G_pos.to_undirected())

print(f"Clustering Pre (Transitivity): {clustering_pre:.4f}")
print()
print(f"Clustering Dur (Transitivity): {clustering_dur:.4f}")
print()
print(f"Clustering Pos (Transitivity): {clustering_pos:.4f}")



```

    Clustering Pre (Transitivity): 0.4942
    
    Clustering Dur (Transitivity): 0.5065
    
    Clustering Pos (Transitivity): 0.5068
    


```python
clustering_pre = nx.transitivity(G_pre)
clustering_dur = nx.transitivity(G_durante)
clustering_pos = nx.transitivity(G_pos)

print(f"Clustering Pre (Transitivity): {clustering_pre:.4f}")
print()
print(f"Clustering Dur (Transitivity): {clustering_dur:.4f}")
print()
print(f"Clustering Pos (Transitivity): {clustering_pos:.4f}")
```

    Clustering Pre (Transitivity): 0.3038
    
    Clustering Dur (Transitivity): 0.3170
    
    Clustering Pos (Transitivity): 0.3085
    


```python
cluster_china_pre = nx.clustering(G_pre,'CHN')
cluster_china_dur = nx.clustering(G_durante,'CHN')
cluster_china_pos = nx.clustering(G_pos,'CHN')
print(f'China Pré: {cluster_china_pre} \nChina Durante: {cluster_china_dur} \nChina Pós: {cluster_china_pos}')
print()
cluster_usa_pre = nx.clustering(G_pre,'USA')
cluster_usa_dur = nx.clustering(G_durante,'USA')
cluster_usa_pos = nx.clustering(G_pos,'USA')
print(f'USA Pré: {cluster_usa_pre} \nUSA Durante: {cluster_usa_dur} \nUSA Pós: {cluster_usa_pos}')
print()
cluster_deu_pre = nx.clustering(G_pre,'DEU')
cluster_deu_dur = nx.clustering(G_durante,'DEU')
cluster_deu_pos = nx.clustering(G_pos,'DEU')
print(f'DEU Pré: {cluster_deu_pre} \nDEU Durante: {cluster_deu_dur} \nDEU Pós: {cluster_deu_pos}')
```

    China Pré: 0.2528732902735562 
    China Durante: 0.26157460343859795 
    China Pós: 0.279345703125
    
    USA Pré: 0.2925983352374735 
    USA Durante: 0.2930787981062204 
    USA Pós: 0.29561004784688993
    
    DEU Pré: 0.2777722735491541 
    DEU Durante: 0.29843929365215055 
    DEU Pós: 0.3046244529332895
    

#### teste -> faz sentido definir bridges desta forma?


```python
def analisar_bridges(G, periodo, thresh_cluster=0.3, thresh_betw=0.02):
    clustering_local = nx.clustering(G)  
    betweenness = nx.betweenness_centrality(G)

    df2 = pd.DataFrame({
        'Nó': list(clustering_local.keys()),
        'Clustering': list(clustering_local.values()),
        'Betweenness': list(betweenness.values())
    })

    df2['Período'] = periodo
    df2['Bridge?'] = (df2['Clustering'] < thresh_cluster) & (df2['Betweenness'] > thresh_betw)

    return df2.sort_values(by='Betweenness', ascending=False)

df_bridges_pre = analisar_bridges(G_pre, 'Pré-Pandemia')
df_bridges_dur = analisar_bridges(G_durante, 'Durante a Pandemia')
df_bridges_pos = analisar_bridges(G_pos, 'Pós-Pandemia')

df_bridges_todos = pd.concat([df_bridges_pre, df_bridges_dur, df_bridges_pos])
df_bridges_todos[df_bridges_todos['Bridge?'] == True]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nó</th>
      <th>Clustering</th>
      <th>Betweenness</th>
      <th>Período</th>
      <th>Bridge?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>CHN</td>
      <td>0.252873</td>
      <td>0.035045</td>
      <td>Pré-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>USA</td>
      <td>0.292598</td>
      <td>0.034894</td>
      <td>Pré-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>64</th>
      <td>DEU</td>
      <td>0.277772</td>
      <td>0.034677</td>
      <td>Pré-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>FRA</td>
      <td>0.287857</td>
      <td>0.026704</td>
      <td>Pré-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NLD</td>
      <td>0.288453</td>
      <td>0.021279</td>
      <td>Pré-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CHN</td>
      <td>0.261575</td>
      <td>0.045976</td>
      <td>Durante a Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>USA</td>
      <td>0.293079</td>
      <td>0.044686</td>
      <td>Durante a Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DEU</td>
      <td>0.298439</td>
      <td>0.025234</td>
      <td>Durante a Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31</th>
      <td>USA</td>
      <td>0.295610</td>
      <td>0.045835</td>
      <td>Pós-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GBR</td>
      <td>0.298247</td>
      <td>0.026175</td>
      <td>Pós-Pandemia</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CHN</td>
      <td>0.279346</td>
      <td>0.021940</td>
      <td>Pós-Pandemia</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### 3.3 - Mapas Temporais


```python
def criar_mapa_temp(networks, coords):
    fig = go.Figure()
    periods = list(networks.keys())
    
    for period_idx, (period_name, G) in enumerate(networks.items()):

        nodes_data = [(n, coords[n]['Longitude'], coords[n]['Latitude'], G.degree(n)) 
                      for n in G.nodes() if n in coords]
        node_sizes = [min(degree * 2, 100) for _, _, _, degree in nodes_data]
        
        # filtrar top 10% das arestas por peso 
        edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)
                              if u in coords and v in coords]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        n_top = int(len(edges_with_weights) * 0.10)
        edges_top = edges_with_weights[:n_top]

        # espessura proporcional ao peso
        if edges_top:
            raw_weights = [w for _, _, w in edges_top]
            edge_widths = [min(0.5 + w / 100_000_000, 5) for w in raw_weights]
            max_width = max(edge_widths) if edge_widths else 5
            
            # categorizar por espessura 
            edge_groups = []
            remaining_edges = list(zip(edges_top, edge_widths))
            
            for threshold, opacity in [(0.7, 0.7), (0.3, 0.5), (0, 0.3)]:
                group = [(u, v, w, ew) for (u, v, w), ew in remaining_edges 
                         if ew > threshold * max_width]
                remaining_edges = [((u, v, w), ew) for (u, v, w), ew in remaining_edges 
                                   if ew <= threshold * max_width]
                edge_groups.append((group, opacity))
        else:
            edge_groups = [([], 0.7), ([], 0.5), ([], 0.3)]
        
        # adicionar 3 traços de arestas
        for edges_list, opacity in edge_groups:
            if edges_list:
                avg_width = sum(e[3] for e in edges_list) / len(edges_list)
                lons = [coord for e in edges_list for coord in 
                       [coords[e[0]]['Longitude'], coords[e[1]]['Longitude'], None]]
                lats = [coord for e in edges_list for coord in 
                       [coords[e[0]]['Latitude'], coords[e[1]]['Latitude'], None]]
            else:
                avg_width = 1
                lons, lats = [], []
            
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=avg_width, color=f'rgba(0, 150, 0, {opacity})'),
                hoverinfo='none', showlegend=False,
                visible=(period_idx == 0)
            ))
        
        # adicionar nós
        fig.add_trace(go.Scattergeo(
            lon=[n[1] for n in nodes_data],
            lat=[n[2] for n in nodes_data],
            mode='markers',
            marker=dict(
                size=[s/5 for s in node_sizes],
                color='red', 
                opacity=0.8
            ),
            text=[f"{n[0]}<br>Conexões: {n[3]}" for n in nodes_data],
            hovertemplate='%{text}<extra></extra>',
            name=period_name,
            visible=(period_idx == 0)
        ))
    
    #configo do mapa e slider
    fig.update_geos(
        projection_type='equirectangular',
        showland=True, landcolor='lightgray',
        showocean=True, oceancolor='lightblue',
        showcountries=True, countrycolor='white'
    )
    
    fig.update_layout(
        title="Evolução das Redes de Exportação de Máscaras Cirúrgicas",
        height=600,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Período: "},
            steps=[dict(
                label=period,
                method="update",
                args=[{"visible": [j // 4 == i for j in range(len(fig.data))]}]
            ) for i, period in enumerate(periods)]
        )]
    )
    
    return fig


coords_df = pd.read_csv('in-nodes-All.csv')  
coords = coords_df.set_index('Label')[['Longitude', 'Latitude']].to_dict('index')

fig = criar_mapa_temp(networks_periodos, coords)
fig.show()
fig.write_html("evolucao_redes_mascaras_periodos.html")
```


<div>                            <div id="611de89c-b139-42b2-ad77-9cdd65cc4f30" class="plotly-graph-div" style="height:600px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("611de89c-b139-42b2-ad77-9cdd65cc4f30")) {                    Plotly.newPlot(                        "611de89c-b139-42b2-ad77-9cdd65cc4f30",                        [{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":5.0},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[22.396428,37.09024,null,22.396428,35.86166,null,35.86166,36.204824,null,23.634501,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1.9576919851099999},"lon":[114.109497,-95.712891,null,114.109497,104.195397,null,104.195397,138.252924,null,-102.552784,-95.712891,null],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,51.165691,null,22.396428,36.204824,null,14.058324,37.09024,null,35.86166,55.378051,null,35.86166,52.132633,null,52.132633,51.165691,null,22.396428,51.165691,null,35.86166,-25.274398,null,35.86166,56.130366,null,20.593684,37.09024,null,35.86166,35.907757,null,22.396428,46.603354,null,35.86166,46.603354,null,37.09024,56.130366,null,51.165691,51.919438,null,22.396428,20.593684,null,22.396428,14.058324,null,51.165691,46.603354,null,12.879721,36.204824,null,22.396428,55.378051,null,18.735693,37.09024,null,35.86166,22.396428,null,51.165691,47.516231,null,35.86166,40.463667,null,35.86166,41.87194,null,22.396428,-0.789275,null,52.132633,50.850346,null,35.86166,50.850346,null,31.791702,40.463667,null,22.396428,52.132633,null,52.132633,46.603354,null,22.396428,-25.274398,null,51.165691,41.87194,null,35.86166,4.210484,null,22.396428,56.130366,null,35.86166,51.919438,null,51.165691,52.132633,null,51.165691,49.817492,null,20.593684,23.424076,null,22.396428,40.463667,null,51.165691,46.818188,null,22.396428,41.87194,null,35.86166,20.593684,null,35.86166,14.058324,null,35.86166,-14.235004,null,35.86166,23.634501,null,35.86166,63.397768,null,35.86166,-35.675147,null,35.86166,23.885942,null,51.165691,55.378051,null,14.058324,51.165691,null,35.86166,23.424076,null,51.919438,51.165691,null,14.058324,36.204824,null,35.86166,61.52401,null,38.963745,51.165691,null,45.943161,51.919438,null,51.165691,40.463667,null,22.396428,23.634501,null,22.396428,22.198745,null,51.165691,63.397768,null,38.963745,37.09024,null,51.165691,50.850346,null,35.86166,12.879721,null,22.396428,12.879721,null,35.86166,15.870032,null,52.132633,55.378051,null,40.463667,46.603354,null,15.870032,37.09024,null,35.86166,1.352083,null,15.870032,51.165691,null,35.86166,33.223191,null,22.396428,-30.559482,null,40.463667,41.87194,null,22.396428,51.919438,null,22.396428,23.685,null,35.86166,-30.559482,null,52.132633,40.463667,null,35.86166,56.26392,null,63.397768,60.472024,null,51.165691,56.26392,null,40.463667,51.165691,null,22.396428,50.850346,null,50.850346,52.132633,null,35.86166,-0.789275,null,40.463667,39.399872,null,38.963745,52.132633,null,52.132633,41.87194,null,22.396428,23.424076,null,22.396428,4.210484,null,20.593684,55.378051,null,45.943161,41.87194,null,22.396428,56.26392,null,-0.789275,36.204824,null,35.86166,-40.900557,null,38.963745,47.411631,null,15.870032,49.817492,null,14.058324,35.907757,null,20.593684,9.081999,null,22.396428,1.352083,null,50.850346,46.603354,null,22.396428,35.907757,null,35.86166,32.427908,null,63.397768,61.92411,null,55.378051,51.165691,null,35.86166,31.046051,null,35.86166,61.92411,null,45.943161,51.165691,null],"line":{"color":"rgba(0, 150, 0, 0.3)","width":0.6472126164455555},"lon":[104.195397,10.451526,null,114.109497,138.252924,null,108.277199,-95.712891,null,104.195397,-3.435973,null,104.195397,5.291266,null,5.291266,10.451526,null,114.109497,10.451526,null,104.195397,133.775136,null,104.195397,-106.346771,null,78.96288,-95.712891,null,104.195397,127.766922,null,114.109497,1.888334,null,104.195397,1.888334,null,-95.712891,-106.346771,null,10.451526,19.145136,null,114.109497,78.96288,null,114.109497,108.277199,null,10.451526,1.888334,null,121.774017,138.252924,null,114.109497,-3.435973,null,-70.162651,-95.712891,null,104.195397,114.109497,null,10.451526,14.550072,null,104.195397,-3.74922,null,104.195397,12.56738,null,114.109497,113.921327,null,5.291266,4.351721,null,104.195397,4.351721,null,-7.09262,-3.74922,null,114.109497,5.291266,null,5.291266,1.888334,null,114.109497,133.775136,null,10.451526,12.56738,null,104.195397,101.975766,null,114.109497,-106.346771,null,104.195397,19.145136,null,10.451526,5.291266,null,10.451526,15.472962,null,78.96288,53.847818,null,114.109497,-3.74922,null,10.451526,8.227512,null,114.109497,12.56738,null,104.195397,78.96288,null,104.195397,108.277199,null,104.195397,-51.92528,null,104.195397,-102.552784,null,104.195397,16.354896,null,104.195397,-71.542969,null,104.195397,45.079162,null,10.451526,-3.435973,null,108.277199,10.451526,null,104.195397,53.847818,null,19.145136,10.451526,null,108.277199,138.252924,null,104.195397,105.318756,null,35.243322,10.451526,null,24.96676,19.145136,null,10.451526,-3.74922,null,114.109497,-102.552784,null,114.109497,113.543873,null,10.451526,16.354896,null,35.243322,-95.712891,null,10.451526,4.351721,null,104.195397,121.774017,null,114.109497,121.774017,null,104.195397,100.992541,null,5.291266,-3.435973,null,-3.74922,1.888334,null,100.992541,-95.712891,null,104.195397,103.819836,null,100.992541,10.451526,null,104.195397,43.679291,null,114.109497,22.937506,null,-3.74922,12.56738,null,114.109497,19.145136,null,114.109497,90.3563,null,104.195397,22.937506,null,5.291266,-3.74922,null,104.195397,9.501785,null,16.354896,8.468946,null,10.451526,9.501785,null,-3.74922,10.451526,null,114.109497,4.351721,null,4.351721,5.291266,null,104.195397,113.921327,null,-3.74922,-8.224454,null,35.243322,5.291266,null,5.291266,12.56738,null,114.109497,53.847818,null,114.109497,101.975766,null,78.96288,-3.435973,null,24.96676,12.56738,null,114.109497,9.501785,null,113.921327,138.252924,null,104.195397,174.885971,null,35.243322,28.369885,null,100.992541,15.472962,null,108.277199,127.766922,null,78.96288,8.675277,null,114.109497,103.819836,null,4.351721,1.888334,null,114.109497,127.766922,null,104.195397,53.688046,null,16.354896,25.748151,null,-3.435973,10.451526,null,104.195397,34.851612,null,104.195397,25.748151,null,24.96676,10.451526,null],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[23.424076,26.02751,26.820553,20.593684,32.427908,33.223191,30.585164,-0.023559,29.31166,21.512583,30.375321,25.354826,23.885942,5.152149,15.552727,-25.274398,35.86166,51.165691,-40.900557,-6.314993,15.870032,37.09024,50.850346,47.516231,46.818188,49.817492,56.26392,40.463667,61.92411,46.603354,55.378051,53.41291,41.87194,49.815273,52.132633,51.919438,39.399872,45.943161,48.669026,46.151241,63.397768,42.733883,43.915886,53.709807,61.52401,-14.235004,-38.416097,-9.189967,-23.442503,-32.522779,-22.328474,-19.015438,23.634501,-35.675147,4.570868,33.0,-11.202692,41.0,9.30769,23.685,-16.290154,4.535277,56.130366,7.539989,7.369722,-2.1646,-0.228021,9.748917,21.521757,35.126413,11.825138,18.735693,28.0,-1.831239,58.595272,9.145,-17.713371,-0.803689,42.315407,7.946527,9.945587,13.443182,39.074208,15.783471,22.396428,15.199999,45.1,47.162494,-0.789275,31.046051,18.109581,36.204824,48.019573,41.20438,12.565679,35.907757,19.85627,33.854721,26.3351,7.873054,55.169438,56.879635,22.198745,31.791702,-18.766947,3.202778,35.937496,21.916221,46.862496,-18.665695,21.00789,-20.348404,4.210484,-22.95764,9.081999,12.865416,60.472024,28.394857,8.537981,12.879721,40.339852,12.862807,14.497401,1.352083,13.794185,44.016521,34.802075,8.619543,10.691803,33.886917,38.963745,-6.369028,48.379433,41.377491,6.42375,14.058324,-30.559482,-13.133897,1.373333,64.963051,41.608635,18.971187,42.546245,-20.904305,-17.679742,4.860416,13.444304,-29.609988,-13.254308,38.969719,-11.6455,47.411631,40.143105,-1.940278,38.861034,25.03428,13.193887,19.3133,-26.522503],"lon":[53.847818,50.55096,30.802498,78.96288,53.688046,43.679291,36.238414,37.906193,47.481766,55.923255,69.345116,51.183884,45.079162,46.199616,48.516388,133.775136,104.195397,10.451526,174.885971,143.95555,100.992541,-95.712891,4.351721,14.550072,8.227512,15.472962,9.501785,-3.74922,25.748151,1.888334,-3.435973,-8.24389,12.56738,6.129583,5.291266,19.145136,-8.224454,24.96676,19.699024,14.995463,16.354896,25.48583,17.679076,27.953389,105.318756,-51.92528,-63.616672,-75.015152,-58.443832,-55.765835,24.684866,29.154857,-102.552784,-71.542969,-74.297333,65.0,17.873887,20.0,2.315834,90.3563,-63.588653,114.727669,-106.346771,-5.54708,12.354722,24.15536,15.827659,-83.753428,-77.781167,33.429859,42.590275,-70.162651,3.0,-78.183406,25.013607,40.489673,178.065033,11.609444,43.356892,-1.023194,-9.696645,-15.310139,21.824312,-90.230759,114.109497,-86.241905,15.2,19.503304,113.921327,34.851612,-77.297508,138.252924,66.923684,74.766098,104.990963,127.766922,102.495496,35.862285,17.228331,80.771797,23.881275,24.603189,113.543873,-7.09262,46.869107,73.22068,14.375416,95.955974,103.846656,35.529562,-10.940835,57.552152,101.975766,18.49041,8.675277,-85.207229,8.468946,84.124008,-80.782127,121.774017,127.510093,30.217636,-14.452362,103.819836,-88.89653,21.005859,38.996815,0.824782,-61.222503,9.537499,35.243322,34.888822,31.16558,64.585262,-66.58973,108.277199,22.937506,27.849332,32.290275,-19.020835,21.745275,-72.285215,1.601554,165.618042,-149.406843,-58.93018,144.793731,28.233608,34.301525,59.556278,43.3333,28.369885,47.576927,29.873888,71.276093,-77.39628,-59.543198,-81.2546,31.465866],"marker":{"color":"red","opacity":0.8,"size":[11.6,2.0,2.4,20.0,2.4,2.0,3.2,2.0,4.4,1.6,5.6,3.6,6.8,0.8,1.2,9.6,20.0,20.0,4.4,0.8,17.6,20.0,15.2,7.6,11.2,14.8,12.8,18.4,6.0,20.0,20.0,7.2,20.0,3.6,20.0,20.0,11.6,15.2,10.8,6.4,20.0,4.8,2.0,3.6,12.0,6.0,2.8,4.0,1.2,0.8,1.2,0.8,8.0,4.4,4.4,0.8,2.0,1.2,0.4,1.2,0.4,0.4,5.6,0.8,0.8,1.6,1.2,1.6,0.8,1.6,0.8,2.0,2.4,2.4,6.8,0.4,1.2,0.4,0.8,2.0,0.4,0.8,5.2,1.6,20.0,1.2,4.8,8.8,6.4,4.4,1.2,14.0,2.4,0.8,2.8,15.2,1.6,1.6,1.2,1.6,8.8,6.4,0.8,4.0,1.6,0.4,1.6,3.2,0.8,1.2,0.4,1.2,8.4,0.8,2.8,1.2,6.4,0.8,1.2,5.2,0.4,1.6,0.8,10.0,1.2,3.6,0.8,0.8,1.2,3.2,20.0,0.8,6.4,0.4,1.6,16.0,11.2,1.2,0.4,0.8,1.6,0.8,0.4,0.4,0.4,0.8,0.4,0.8,0.4,0.8,0.4,2.0,0.8,0.4,0.4,0.4,0.4,0.4,0.4]},"mode":"markers","name":"Pr\u00e9-Pandemia","text":["ARE\u003cbr\u003eConex\u00f5es: 29","BHR\u003cbr\u003eConex\u00f5es: 5","EGY\u003cbr\u003eConex\u00f5es: 6","IND\u003cbr\u003eConex\u00f5es: 78","IRN\u003cbr\u003eConex\u00f5es: 6","IRQ\u003cbr\u003eConex\u00f5es: 5","JOR\u003cbr\u003eConex\u00f5es: 8","KEN\u003cbr\u003eConex\u00f5es: 5","KWT\u003cbr\u003eConex\u00f5es: 11","OMN\u003cbr\u003eConex\u00f5es: 4","PAK\u003cbr\u003eConex\u00f5es: 14","QAT\u003cbr\u003eConex\u00f5es: 9","SAU\u003cbr\u003eConex\u00f5es: 17","SOM\u003cbr\u003eConex\u00f5es: 2","YEM\u003cbr\u003eConex\u00f5es: 3","AUS\u003cbr\u003eConex\u00f5es: 24","CHN\u003cbr\u003eConex\u00f5es: 152","DEU\u003cbr\u003eConex\u00f5es: 107","NZL\u003cbr\u003eConex\u00f5es: 11","PNG\u003cbr\u003eConex\u00f5es: 2","THA\u003cbr\u003eConex\u00f5es: 44","USA\u003cbr\u003eConex\u00f5es: 88","BEL\u003cbr\u003eConex\u00f5es: 38","AUT\u003cbr\u003eConex\u00f5es: 19","CHE\u003cbr\u003eConex\u00f5es: 28","CZE\u003cbr\u003eConex\u00f5es: 37","DNK\u003cbr\u003eConex\u00f5es: 32","ESP\u003cbr\u003eConex\u00f5es: 46","FIN\u003cbr\u003eConex\u00f5es: 15","FRA\u003cbr\u003eConex\u00f5es: 69","GBR\u003cbr\u003eConex\u00f5es: 54","IRL\u003cbr\u003eConex\u00f5es: 18","ITA\u003cbr\u003eConex\u00f5es: 56","LUX\u003cbr\u003eConex\u00f5es: 9","NLD\u003cbr\u003eConex\u00f5es: 67","POL\u003cbr\u003eConex\u00f5es: 56","PRT\u003cbr\u003eConex\u00f5es: 29","ROU\u003cbr\u003eConex\u00f5es: 38","SVK\u003cbr\u003eConex\u00f5es: 27","SVN\u003cbr\u003eConex\u00f5es: 16","SWE\u003cbr\u003eConex\u00f5es: 50","BGR\u003cbr\u003eConex\u00f5es: 12","BIH\u003cbr\u003eConex\u00f5es: 5","BLR\u003cbr\u003eConex\u00f5es: 9","RUS\u003cbr\u003eConex\u00f5es: 30","BRA\u003cbr\u003eConex\u00f5es: 15","ARG\u003cbr\u003eConex\u00f5es: 7","PER\u003cbr\u003eConex\u00f5es: 10","PRY\u003cbr\u003eConex\u00f5es: 3","URY\u003cbr\u003eConex\u00f5es: 2","BWA\u003cbr\u003eConex\u00f5es: 3","ZWE\u003cbr\u003eConex\u00f5es: 2","MEX\u003cbr\u003eConex\u00f5es: 20","CHL\u003cbr\u003eConex\u00f5es: 11","COL\u003cbr\u003eConex\u00f5es: 11","AFG\u003cbr\u003eConex\u00f5es: 2","AGO\u003cbr\u003eConex\u00f5es: 5","ALB\u003cbr\u003eConex\u00f5es: 3","BEN\u003cbr\u003eConex\u00f5es: 1","BGD\u003cbr\u003eConex\u00f5es: 3","BOL\u003cbr\u003eConex\u00f5es: 1","BRN\u003cbr\u003eConex\u00f5es: 1","CAN\u003cbr\u003eConex\u00f5es: 14","CIV\u003cbr\u003eConex\u00f5es: 2","CMR\u003cbr\u003eConex\u00f5es: 2","COD\u003cbr\u003eConex\u00f5es: 4","COG\u003cbr\u003eConex\u00f5es: 3","CRI\u003cbr\u003eConex\u00f5es: 4","CUB\u003cbr\u003eConex\u00f5es: 2","CYP\u003cbr\u003eConex\u00f5es: 4","DJI\u003cbr\u003eConex\u00f5es: 2","DOM\u003cbr\u003eConex\u00f5es: 5","DZA\u003cbr\u003eConex\u00f5es: 6","ECU\u003cbr\u003eConex\u00f5es: 6","EST\u003cbr\u003eConex\u00f5es: 17","ETH\u003cbr\u003eConex\u00f5es: 1","FJI\u003cbr\u003eConex\u00f5es: 3","GAB\u003cbr\u003eConex\u00f5es: 1","GEO\u003cbr\u003eConex\u00f5es: 2","GHA\u003cbr\u003eConex\u00f5es: 5","GIN\u003cbr\u003eConex\u00f5es: 1","GMB\u003cbr\u003eConex\u00f5es: 2","GRC\u003cbr\u003eConex\u00f5es: 13","GTM\u003cbr\u003eConex\u00f5es: 4","HKG\u003cbr\u003eConex\u00f5es: 102","HND\u003cbr\u003eConex\u00f5es: 3","HRV\u003cbr\u003eConex\u00f5es: 12","HUN\u003cbr\u003eConex\u00f5es: 22","IDN\u003cbr\u003eConex\u00f5es: 16","ISR\u003cbr\u003eConex\u00f5es: 11","JAM\u003cbr\u003eConex\u00f5es: 3","JPN\u003cbr\u003eConex\u00f5es: 35","KAZ\u003cbr\u003eConex\u00f5es: 6","KGZ\u003cbr\u003eConex\u00f5es: 2","KHM\u003cbr\u003eConex\u00f5es: 7","KOR\u003cbr\u003eConex\u00f5es: 38","LAO\u003cbr\u003eConex\u00f5es: 4","LBN\u003cbr\u003eConex\u00f5es: 4","LBY\u003cbr\u003eConex\u00f5es: 3","LKA\u003cbr\u003eConex\u00f5es: 4","LTU\u003cbr\u003eConex\u00f5es: 22","LVA\u003cbr\u003eConex\u00f5es: 16","MAC\u003cbr\u003eConex\u00f5es: 2","MAR\u003cbr\u003eConex\u00f5es: 10","MDG\u003cbr\u003eConex\u00f5es: 4","MDV\u003cbr\u003eConex\u00f5es: 1","MLT\u003cbr\u003eConex\u00f5es: 4","MMR\u003cbr\u003eConex\u00f5es: 8","MNG\u003cbr\u003eConex\u00f5es: 2","MOZ\u003cbr\u003eConex\u00f5es: 3","MRT\u003cbr\u003eConex\u00f5es: 1","MUS\u003cbr\u003eConex\u00f5es: 3","MYS\u003cbr\u003eConex\u00f5es: 21","NAM\u003cbr\u003eConex\u00f5es: 2","NGA\u003cbr\u003eConex\u00f5es: 7","NIC\u003cbr\u003eConex\u00f5es: 3","NOR\u003cbr\u003eConex\u00f5es: 16","NPL\u003cbr\u003eConex\u00f5es: 2","PAN\u003cbr\u003eConex\u00f5es: 3","PHL\u003cbr\u003eConex\u00f5es: 13","PRK\u003cbr\u003eConex\u00f5es: 1","SDN\u003cbr\u003eConex\u00f5es: 4","SEN\u003cbr\u003eConex\u00f5es: 2","SGP\u003cbr\u003eConex\u00f5es: 25","SLV\u003cbr\u003eConex\u00f5es: 3","SRB\u003cbr\u003eConex\u00f5es: 9","SYR\u003cbr\u003eConex\u00f5es: 2","TGO\u003cbr\u003eConex\u00f5es: 2","TTO\u003cbr\u003eConex\u00f5es: 3","TUN\u003cbr\u003eConex\u00f5es: 8","TUR\u003cbr\u003eConex\u00f5es: 77","TZA\u003cbr\u003eConex\u00f5es: 2","UKR\u003cbr\u003eConex\u00f5es: 16","UZB\u003cbr\u003eConex\u00f5es: 1","VEN\u003cbr\u003eConex\u00f5es: 4","VNM\u003cbr\u003eConex\u00f5es: 40","ZAF\u003cbr\u003eConex\u00f5es: 28","ZMB\u003cbr\u003eConex\u00f5es: 3","UGA\u003cbr\u003eConex\u00f5es: 1","ISL\u003cbr\u003eConex\u00f5es: 2","MKD\u003cbr\u003eConex\u00f5es: 4","HTI\u003cbr\u003eConex\u00f5es: 2","AND\u003cbr\u003eConex\u00f5es: 1","NCL\u003cbr\u003eConex\u00f5es: 1","PYF\u003cbr\u003eConex\u00f5es: 1","GUY\u003cbr\u003eConex\u00f5es: 2","GUM\u003cbr\u003eConex\u00f5es: 1","LSO\u003cbr\u003eConex\u00f5es: 2","MWI\u003cbr\u003eConex\u00f5es: 1","TKM\u003cbr\u003eConex\u00f5es: 2","COM\u003cbr\u003eConex\u00f5es: 1","MDA\u003cbr\u003eConex\u00f5es: 5","AZE\u003cbr\u003eConex\u00f5es: 2","RWA\u003cbr\u003eConex\u00f5es: 1","TJK\u003cbr\u003eConex\u00f5es: 1","BHS\u003cbr\u003eConex\u00f5es: 1","BRB\u003cbr\u003eConex\u00f5es: 1","CYM\u003cbr\u003eConex\u00f5es: 1","SWZ\u003cbr\u003eConex\u00f5es: 1"],"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":5.0},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,36.204824,null,35.86166,51.165691,null,35.86166,55.378051,null,35.86166,52.132633,null,35.86166,46.603354,null,35.86166,56.130366,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":2.0998692592733335},"lon":[104.195397,138.252924,null,104.195397,10.451526,null,104.195397,-3.435973,null,104.195397,5.291266,null,104.195397,1.888334,null,104.195397,-106.346771,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,-25.274398,null,37.09024,23.634501,null,35.86166,23.634501,null,35.86166,40.463667,null,35.86166,35.907757,null,35.86166,-0.789275,null,23.634501,37.09024,null,37.09024,56.130366,null,52.132633,51.165691,null,35.86166,41.87194,null,35.86166,61.52401,null,35.86166,15.870032,null,35.86166,4.210484,null,14.058324,37.09024,null,35.86166,-35.675147,null,35.86166,12.879721,null,35.86166,23.424076,null,35.86166,14.058324,null,35.86166,50.850346,null,35.86166,23.885942,null,35.86166,51.919438,null,35.86166,-14.235004,null,35.86166,-9.189967,null,51.165691,46.603354,null,51.165691,51.919438,null,35.86166,1.352083,null,35.86166,63.397768,null,35.86166,-30.559482,null,50.850346,46.603354,null,51.919438,51.165691,null,51.165691,47.516231,null,20.593684,37.09024,null,35.86166,4.570868,null,35.86166,31.046051,null,35.86166,21.916221,null,35.86166,20.593684,null,52.132633,46.603354,null,35.86166,46.818188,null,35.86166,8.537981,null,52.132633,50.850346,null,31.791702,40.463667,null,20.593684,23.424076,null,56.130366,37.09024,null,38.963745,37.09024,null,22.396428,37.09024,null,18.735693,37.09024,null,51.165691,52.132633,null,38.963745,51.165691,null,12.879721,36.204824,null,35.86166,56.26392,null,15.870032,37.09024,null,51.165691,49.817492,null,14.058324,36.204824,null,51.165691,41.87194,null,51.165691,46.818188,null,35.86166,61.92411,null,35.86166,33.223191,null,35.86166,39.399872,null,35.86166,39.074208,null,39.399872,40.463667,null,35.86166,26.820553,null,12.565679,37.09024,null,51.165691,40.463667,null,40.463667,46.603354,null,51.919438,56.26392,null,35.86166,-40.900557,null,14.058324,46.603354,null,35.86166,22.396428,null,-18.766947,46.603354,null,35.86166,45.943161,null,35.86166,49.817492,null,35.86166,-1.831239,null,49.817492,51.165691,null,14.058324,51.165691,null,35.86166,47.516231,null,35.86166,29.31166,null,33.886917,46.603354,null,38.963745,52.132633,null,45.943161,51.919438,null,35.86166,60.472024,null,38.963745,46.603354,null,40.463667,41.87194,null,35.86166,46.151241,null,40.463667,39.399872,null,50.850346,52.132633,null,35.86166,15.783471,null,22.396428,35.86166,null,35.86166,53.41291,null,55.378051,53.41291,null,46.603354,41.87194,null,51.165691,55.378051,null,37.09024,50.850346,null,35.86166,-38.416097,null,55.378051,51.165691,null,35.86166,48.379433,null,52.132633,41.87194,null,51.165691,50.850346,null,35.86166,9.081999,null,35.86166,47.162494,null,35.86166,7.946527,null,39.399872,46.603354,null,35.907757,37.09024,null,51.919438,49.817492,null,63.397768,60.472024,null,46.603354,51.165691,null,46.603354,40.463667,null,51.165691,63.397768,null,37.09024,-25.274398,null,35.907757,36.204824,null,55.378051,56.26392,null,35.86166,55.169438,null,52.132633,40.463667,null,52.132633,37.09024,null,51.919438,48.669026,null,63.397768,61.92411,null,35.86166,25.354826,null,45.943161,51.165691,null,40.463667,51.165691,null,38.963745,47.411631,null,35.907757,35.86166,null,41.87194,46.603354,null,35.86166,12.565679,null,50.850346,51.165691,null,35.86166,13.794185,null,55.378051,46.603354,null],"line":{"color":"rgba(0, 150, 0, 0.3)","width":0.6611188820987199},"lon":[104.195397,133.775136,null,-95.712891,-102.552784,null,104.195397,-102.552784,null,104.195397,-3.74922,null,104.195397,127.766922,null,104.195397,113.921327,null,-102.552784,-95.712891,null,-95.712891,-106.346771,null,5.291266,10.451526,null,104.195397,12.56738,null,104.195397,105.318756,null,104.195397,100.992541,null,104.195397,101.975766,null,108.277199,-95.712891,null,104.195397,-71.542969,null,104.195397,121.774017,null,104.195397,53.847818,null,104.195397,108.277199,null,104.195397,4.351721,null,104.195397,45.079162,null,104.195397,19.145136,null,104.195397,-51.92528,null,104.195397,-75.015152,null,10.451526,1.888334,null,10.451526,19.145136,null,104.195397,103.819836,null,104.195397,16.354896,null,104.195397,22.937506,null,4.351721,1.888334,null,19.145136,10.451526,null,10.451526,14.550072,null,78.96288,-95.712891,null,104.195397,-74.297333,null,104.195397,34.851612,null,104.195397,95.955974,null,104.195397,78.96288,null,5.291266,1.888334,null,104.195397,8.227512,null,104.195397,-80.782127,null,5.291266,4.351721,null,-7.09262,-3.74922,null,78.96288,53.847818,null,-106.346771,-95.712891,null,35.243322,-95.712891,null,114.109497,-95.712891,null,-70.162651,-95.712891,null,10.451526,5.291266,null,35.243322,10.451526,null,121.774017,138.252924,null,104.195397,9.501785,null,100.992541,-95.712891,null,10.451526,15.472962,null,108.277199,138.252924,null,10.451526,12.56738,null,10.451526,8.227512,null,104.195397,25.748151,null,104.195397,43.679291,null,104.195397,-8.224454,null,104.195397,21.824312,null,-8.224454,-3.74922,null,104.195397,30.802498,null,104.990963,-95.712891,null,10.451526,-3.74922,null,-3.74922,1.888334,null,19.145136,9.501785,null,104.195397,174.885971,null,108.277199,1.888334,null,104.195397,114.109497,null,46.869107,1.888334,null,104.195397,24.96676,null,104.195397,15.472962,null,104.195397,-78.183406,null,15.472962,10.451526,null,108.277199,10.451526,null,104.195397,14.550072,null,104.195397,47.481766,null,9.537499,1.888334,null,35.243322,5.291266,null,24.96676,19.145136,null,104.195397,8.468946,null,35.243322,1.888334,null,-3.74922,12.56738,null,104.195397,14.995463,null,-3.74922,-8.224454,null,4.351721,5.291266,null,104.195397,-90.230759,null,114.109497,104.195397,null,104.195397,-8.24389,null,-3.435973,-8.24389,null,1.888334,12.56738,null,10.451526,-3.435973,null,-95.712891,4.351721,null,104.195397,-63.616672,null,-3.435973,10.451526,null,104.195397,31.16558,null,5.291266,12.56738,null,10.451526,4.351721,null,104.195397,8.675277,null,104.195397,19.503304,null,104.195397,-1.023194,null,-8.224454,1.888334,null,127.766922,-95.712891,null,19.145136,15.472962,null,16.354896,8.468946,null,1.888334,10.451526,null,1.888334,-3.74922,null,10.451526,16.354896,null,-95.712891,133.775136,null,127.766922,138.252924,null,-3.435973,9.501785,null,104.195397,23.881275,null,5.291266,-3.74922,null,5.291266,-95.712891,null,19.145136,19.699024,null,16.354896,25.748151,null,104.195397,51.183884,null,24.96676,10.451526,null,-3.74922,10.451526,null,35.243322,28.369885,null,127.766922,104.195397,null,12.56738,1.888334,null,104.195397,104.990963,null,4.351721,10.451526,null,104.195397,-88.89653,null,-3.435973,1.888334,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[-38.416097,-32.522779,40.069099,61.52401,-25.274398,35.86166,-17.713371,22.396428,20.593684,-40.900557,-6.314993,15.870032,37.09024,50.850346,47.516231,46.818188,49.817492,51.165691,56.26392,28.0,40.463667,61.92411,46.603354,55.378051,39.074208,53.41291,41.87194,49.815273,56.879635,52.132633,51.919438,39.399872,46.151241,63.397768,33.886917,-30.559482,42.733883,45.1,45.943161,44.016521,26.02751,43.915886,53.709807,38.963745,48.379433,-16.290154,-23.442503,6.42375,-14.235004,-35.675147,4.570868,23.634501,56.130366,23.885942,-9.189967,33.0,-11.202692,41.0,23.424076,40.143105,9.30769,12.238333,23.685,13.193887,4.535277,-22.328474,7.539989,7.369722,-2.1646,-0.228021,16.5388,9.748917,21.521757,35.126413,11.825138,18.735693,-1.831239,26.820553,58.595272,9.145,-0.803689,42.315407,7.946527,9.945587,13.443182,15.783471,4.860416,15.199999,18.971187,47.162494,-0.789275,32.427908,33.223191,64.963051,31.046051,18.109581,30.585164,36.204824,48.019573,-0.023559,41.20438,12.565679,35.907757,29.31166,19.85627,33.854721,6.428055,26.3351,7.873054,55.169438,22.198745,31.791702,47.411631,-18.766947,3.202778,41.608635,17.570692,35.937496,21.916221,46.862496,-18.665695,21.00789,-20.348404,-13.254308,4.210484,-22.95764,-20.904305,17.607789,9.081999,12.865416,60.472024,28.394857,21.512583,30.375321,8.537981,12.879721,40.339852,31.952162,-17.679742,25.354826,-1.940278,12.862807,14.497401,1.352083,8.460555,13.794185,5.152149,7.862684,3.919305,48.669026,34.802075,15.454166,8.619543,38.861034,38.969719,-8.874217,10.691803,-6.369028,1.373333,41.377491,14.058324,15.552727,-13.133897,-19.015438,42.546245,42.708678,12.262776,-11.6455,25.03428,17.189877,19.3133,15.414999,12.984305,-29.609988,-26.522503],"lon":[-63.616672,-55.765835,45.038189,105.318756,133.775136,104.195397,178.065033,114.109497,78.96288,174.885971,143.95555,100.992541,-95.712891,4.351721,14.550072,8.227512,15.472962,10.451526,9.501785,3.0,-3.74922,25.748151,1.888334,-3.435973,21.824312,-8.24389,12.56738,6.129583,24.603189,5.291266,19.145136,-8.224454,14.995463,16.354896,9.537499,22.937506,25.48583,15.2,24.96676,21.005859,50.55096,17.679076,27.953389,35.243322,31.16558,-63.588653,-58.443832,-66.58973,-51.92528,-71.542969,-74.297333,-102.552784,-106.346771,45.079162,-75.015152,65.0,17.873887,20.0,53.847818,47.576927,2.315834,-1.561593,90.3563,-59.543198,114.727669,24.684866,-5.54708,12.354722,24.15536,15.827659,-23.0418,-83.753428,-77.781167,33.429859,42.590275,-70.162651,-78.183406,30.802498,25.013607,40.489673,11.609444,43.356892,-1.023194,-9.696645,-15.310139,-90.230759,-58.93018,-86.241905,-72.285215,19.503304,113.921327,53.688046,43.679291,-19.020835,34.851612,-77.297508,36.238414,138.252924,66.923684,37.906193,74.766098,104.990963,127.766922,47.481766,102.495496,35.862285,-9.429499,17.228331,80.771797,23.881275,113.543873,-7.09262,28.369885,46.869107,73.22068,21.745275,-3.996166,14.375416,95.955974,103.846656,35.529562,-10.940835,57.552152,34.301525,101.975766,18.49041,165.618042,8.081666,8.675277,-85.207229,8.468946,84.124008,55.923255,69.345116,-80.782127,121.774017,127.510093,35.233154,-149.406843,51.183884,29.873888,30.217636,-14.452362,103.819836,-11.779889,-88.89653,46.199616,30.217636,-56.027783,19.699024,38.996815,18.732207,0.824782,71.276093,59.556278,125.727,-61.222503,34.888822,32.290275,64.585262,108.277199,48.516388,27.849332,29.154857,1.601554,19.37439,-61.604171,43.3333,-77.39628,-88.49765,-81.2546,-61.370976,-61.287228,28.233608,31.465866],"marker":{"color":"red","opacity":0.8,"size":[4.4,2.0,0.8,10.8,11.2,20.0,1.2,20.0,20.0,4.4,0.8,14.0,20.0,20.0,8.4,11.2,19.2,20.0,15.2,2.0,20.0,7.6,20.0,20.0,9.2,9.2,20.0,7.2,6.4,20.0,20.0,14.4,9.2,17.2,6.0,9.2,10.4,8.0,16.4,9.2,0.8,3.2,4.4,20.0,6.0,2.0,2.8,1.2,9.2,5.6,6.0,10.0,10.0,6.0,3.6,0.8,1.2,1.6,5.6,0.8,0.4,0.8,2.0,0.8,0.8,0.8,1.2,0.4,0.4,0.4,0.8,1.2,0.4,2.8,0.8,3.2,2.4,2.4,6.8,0.8,0.8,4.0,2.4,0.8,0.4,4.0,0.8,2.4,1.2,13.2,8.4,1.2,2.0,2.4,5.6,0.8,1.6,16.0,1.6,2.8,1.2,6.4,17.6,2.0,1.6,0.4,0.4,0.8,1.2,12.0,1.2,6.4,2.8,1.6,0.4,2.4,0.8,1.6,3.6,0.8,1.2,0.8,0.8,0.4,12.4,0.8,0.8,0.4,1.6,3.2,5.2,0.8,1.2,4.4,2.4,7.2,0.4,0.4,0.8,2.0,0.8,1.2,1.6,14.8,0.8,3.6,0.4,0.4,0.8,12.4,0.8,0.4,1.2,0.8,0.8,0.4,1.2,1.2,0.8,2.0,19.2,0.4,1.6,0.8,0.4,0.8,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]},"mode":"markers","name":"Durante Pandemia","text":["ARG\u003cbr\u003eConex\u00f5es: 11","URY\u003cbr\u003eConex\u00f5es: 5","ARM\u003cbr\u003eConex\u00f5es: 2","RUS\u003cbr\u003eConex\u00f5es: 27","AUS\u003cbr\u003eConex\u00f5es: 28","CHN\u003cbr\u003eConex\u00f5es: 197","FJI\u003cbr\u003eConex\u00f5es: 3","HKG\u003cbr\u003eConex\u00f5es: 76","IND\u003cbr\u003eConex\u00f5es: 66","NZL\u003cbr\u003eConex\u00f5es: 11","PNG\u003cbr\u003eConex\u00f5es: 2","THA\u003cbr\u003eConex\u00f5es: 35","USA\u003cbr\u003eConex\u00f5es: 109","BEL\u003cbr\u003eConex\u00f5es: 52","AUT\u003cbr\u003eConex\u00f5es: 21","CHE\u003cbr\u003eConex\u00f5es: 28","CZE\u003cbr\u003eConex\u00f5es: 48","DEU\u003cbr\u003eConex\u00f5es: 114","DNK\u003cbr\u003eConex\u00f5es: 38","DZA\u003cbr\u003eConex\u00f5es: 5","ESP\u003cbr\u003eConex\u00f5es: 51","FIN\u003cbr\u003eConex\u00f5es: 19","FRA\u003cbr\u003eConex\u00f5es: 87","GBR\u003cbr\u003eConex\u00f5es: 68","GRC\u003cbr\u003eConex\u00f5es: 23","IRL\u003cbr\u003eConex\u00f5es: 23","ITA\u003cbr\u003eConex\u00f5es: 64","LUX\u003cbr\u003eConex\u00f5es: 18","LVA\u003cbr\u003eConex\u00f5es: 16","NLD\u003cbr\u003eConex\u00f5es: 82","POL\u003cbr\u003eConex\u00f5es: 59","PRT\u003cbr\u003eConex\u00f5es: 36","SVN\u003cbr\u003eConex\u00f5es: 23","SWE\u003cbr\u003eConex\u00f5es: 43","TUN\u003cbr\u003eConex\u00f5es: 15","ZAF\u003cbr\u003eConex\u00f5es: 23","BGR\u003cbr\u003eConex\u00f5es: 26","HRV\u003cbr\u003eConex\u00f5es: 20","ROU\u003cbr\u003eConex\u00f5es: 41","SRB\u003cbr\u003eConex\u00f5es: 23","BHR\u003cbr\u003eConex\u00f5es: 2","BIH\u003cbr\u003eConex\u00f5es: 8","BLR\u003cbr\u003eConex\u00f5es: 11","TUR\u003cbr\u003eConex\u00f5es: 87","UKR\u003cbr\u003eConex\u00f5es: 15","BOL\u003cbr\u003eConex\u00f5es: 5","PRY\u003cbr\u003eConex\u00f5es: 7","VEN\u003cbr\u003eConex\u00f5es: 3","BRA\u003cbr\u003eConex\u00f5es: 23","CHL\u003cbr\u003eConex\u00f5es: 14","COL\u003cbr\u003eConex\u00f5es: 15","MEX\u003cbr\u003eConex\u00f5es: 25","CAN\u003cbr\u003eConex\u00f5es: 25","SAU\u003cbr\u003eConex\u00f5es: 15","PER\u003cbr\u003eConex\u00f5es: 9","AFG\u003cbr\u003eConex\u00f5es: 2","AGO\u003cbr\u003eConex\u00f5es: 3","ALB\u003cbr\u003eConex\u00f5es: 4","ARE\u003cbr\u003eConex\u00f5es: 14","AZE\u003cbr\u003eConex\u00f5es: 2","BEN\u003cbr\u003eConex\u00f5es: 1","BFA\u003cbr\u003eConex\u00f5es: 2","BGD\u003cbr\u003eConex\u00f5es: 5","BRB\u003cbr\u003eConex\u00f5es: 2","BRN\u003cbr\u003eConex\u00f5es: 2","BWA\u003cbr\u003eConex\u00f5es: 2","CIV\u003cbr\u003eConex\u00f5es: 3","CMR\u003cbr\u003eConex\u00f5es: 1","COD\u003cbr\u003eConex\u00f5es: 1","COG\u003cbr\u003eConex\u00f5es: 1","CPV\u003cbr\u003eConex\u00f5es: 2","CRI\u003cbr\u003eConex\u00f5es: 3","CUB\u003cbr\u003eConex\u00f5es: 1","CYP\u003cbr\u003eConex\u00f5es: 7","DJI\u003cbr\u003eConex\u00f5es: 2","DOM\u003cbr\u003eConex\u00f5es: 8","ECU\u003cbr\u003eConex\u00f5es: 6","EGY\u003cbr\u003eConex\u00f5es: 6","EST\u003cbr\u003eConex\u00f5es: 17","ETH\u003cbr\u003eConex\u00f5es: 2","GAB\u003cbr\u003eConex\u00f5es: 2","GEO\u003cbr\u003eConex\u00f5es: 10","GHA\u003cbr\u003eConex\u00f5es: 6","GIN\u003cbr\u003eConex\u00f5es: 2","GMB\u003cbr\u003eConex\u00f5es: 1","GTM\u003cbr\u003eConex\u00f5es: 10","GUY\u003cbr\u003eConex\u00f5es: 2","HND\u003cbr\u003eConex\u00f5es: 6","HTI\u003cbr\u003eConex\u00f5es: 3","HUN\u003cbr\u003eConex\u00f5es: 33","IDN\u003cbr\u003eConex\u00f5es: 21","IRN\u003cbr\u003eConex\u00f5es: 3","IRQ\u003cbr\u003eConex\u00f5es: 5","ISL\u003cbr\u003eConex\u00f5es: 6","ISR\u003cbr\u003eConex\u00f5es: 14","JAM\u003cbr\u003eConex\u00f5es: 2","JOR\u003cbr\u003eConex\u00f5es: 4","JPN\u003cbr\u003eConex\u00f5es: 40","KAZ\u003cbr\u003eConex\u00f5es: 4","KEN\u003cbr\u003eConex\u00f5es: 7","KGZ\u003cbr\u003eConex\u00f5es: 3","KHM\u003cbr\u003eConex\u00f5es: 16","KOR\u003cbr\u003eConex\u00f5es: 44","KWT\u003cbr\u003eConex\u00f5es: 5","LAO\u003cbr\u003eConex\u00f5es: 4","LBN\u003cbr\u003eConex\u00f5es: 1","LBR\u003cbr\u003eConex\u00f5es: 1","LBY\u003cbr\u003eConex\u00f5es: 2","LKA\u003cbr\u003eConex\u00f5es: 3","LTU\u003cbr\u003eConex\u00f5es: 30","MAC\u003cbr\u003eConex\u00f5es: 3","MAR\u003cbr\u003eConex\u00f5es: 16","MDA\u003cbr\u003eConex\u00f5es: 7","MDG\u003cbr\u003eConex\u00f5es: 4","MDV\u003cbr\u003eConex\u00f5es: 1","MKD\u003cbr\u003eConex\u00f5es: 6","MLI\u003cbr\u003eConex\u00f5es: 2","MLT\u003cbr\u003eConex\u00f5es: 4","MMR\u003cbr\u003eConex\u00f5es: 9","MNG\u003cbr\u003eConex\u00f5es: 2","MOZ\u003cbr\u003eConex\u00f5es: 3","MRT\u003cbr\u003eConex\u00f5es: 2","MUS\u003cbr\u003eConex\u00f5es: 2","MWI\u003cbr\u003eConex\u00f5es: 1","MYS\u003cbr\u003eConex\u00f5es: 31","NAM\u003cbr\u003eConex\u00f5es: 2","NCL\u003cbr\u003eConex\u00f5es: 2","NER\u003cbr\u003eConex\u00f5es: 1","NGA\u003cbr\u003eConex\u00f5es: 4","NIC\u003cbr\u003eConex\u00f5es: 8","NOR\u003cbr\u003eConex\u00f5es: 13","NPL\u003cbr\u003eConex\u00f5es: 2","OMN\u003cbr\u003eConex\u00f5es: 3","PAK\u003cbr\u003eConex\u00f5es: 11","PAN\u003cbr\u003eConex\u00f5es: 6","PHL\u003cbr\u003eConex\u00f5es: 18","PRK\u003cbr\u003eConex\u00f5es: 1","PSE\u003cbr\u003eConex\u00f5es: 1","PYF\u003cbr\u003eConex\u00f5es: 2","QAT\u003cbr\u003eConex\u00f5es: 5","RWA\u003cbr\u003eConex\u00f5es: 2","SDN\u003cbr\u003eConex\u00f5es: 3","SEN\u003cbr\u003eConex\u00f5es: 4","SGP\u003cbr\u003eConex\u00f5es: 37","SLE\u003cbr\u003eConex\u00f5es: 2","SLV\u003cbr\u003eConex\u00f5es: 9","SOM\u003cbr\u003eConex\u00f5es: 1","SSD\u003cbr\u003eConex\u00f5es: 1","SUR\u003cbr\u003eConex\u00f5es: 2","SVK\u003cbr\u003eConex\u00f5es: 31","SYR\u003cbr\u003eConex\u00f5es: 2","TCD\u003cbr\u003eConex\u00f5es: 1","TGO\u003cbr\u003eConex\u00f5es: 3","TJK\u003cbr\u003eConex\u00f5es: 2","TKM\u003cbr\u003eConex\u00f5es: 2","TLS\u003cbr\u003eConex\u00f5es: 1","TTO\u003cbr\u003eConex\u00f5es: 3","TZA\u003cbr\u003eConex\u00f5es: 3","UGA\u003cbr\u003eConex\u00f5es: 2","UZB\u003cbr\u003eConex\u00f5es: 5","VNM\u003cbr\u003eConex\u00f5es: 48","YEM\u003cbr\u003eConex\u00f5es: 1","ZMB\u003cbr\u003eConex\u00f5es: 4","ZWE\u003cbr\u003eConex\u00f5es: 2","AND\u003cbr\u003eConex\u00f5es: 1","MNE\u003cbr\u003eConex\u00f5es: 2","GRD\u003cbr\u003eConex\u00f5es: 1","COM\u003cbr\u003eConex\u00f5es: 1","BHS\u003cbr\u003eConex\u00f5es: 1","BLZ\u003cbr\u003eConex\u00f5es: 1","CYM\u003cbr\u003eConex\u00f5es: 1","DMA\u003cbr\u003eConex\u00f5es: 1","VCT\u003cbr\u003eConex\u00f5es: 1","LSO\u003cbr\u003eConex\u00f5es: 1","SWZ\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":5.0},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,36.204824,null,23.634501,37.09024,null,37.09024,23.634501,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":2.0662747050333334},"lon":[104.195397,138.252924,null,-102.552784,-95.712891,null,-95.712891,-102.552784,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,51.165691,null,52.132633,51.165691,null,35.86166,52.132633,null,35.86166,55.378051,null,35.86166,-25.274398,null,35.86166,56.130366,null,35.86166,15.870032,null,37.09024,56.130366,null,35.86166,23.634501,null,35.86166,46.603354,null,35.86166,12.879721,null,52.132633,46.603354,null,35.86166,35.907757,null,35.86166,-0.789275,null,52.132633,50.850346,null,20.593684,23.424076,null,35.86166,-35.675147,null,51.919438,51.165691,null,35.86166,40.463667,null,51.165691,46.603354,null,-18.766947,46.603354,null,35.86166,4.210484,null,35.86166,51.919438,null,51.165691,51.919438,null,52.132633,51.919438,null,35.86166,41.87194,null,14.058324,37.09024,null,38.963745,37.09024,null,35.86166,-14.235004,null,18.735693,37.09024,null,35.86166,14.058324,null,35.86166,50.850346,null,51.165691,47.516231,null,35.86166,20.593684,null,30.375321,37.09024,null,40.463667,46.603354,null,20.593684,37.09024,null,35.86166,23.424076,null,35.86166,23.885942,null,35.86166,61.52401,null,15.870032,37.09024,null,35.86166,63.397768,null,56.130366,37.09024,null,51.165691,46.818188,null,51.165691,41.87194,null,35.86166,-9.189967,null,38.963745,51.165691,null,51.165691,49.817492,null,51.165691,40.463667,null,49.817492,51.165691,null,12.565679,37.09024,null,56.26392,51.165691,null,39.399872,40.463667,null,35.86166,-30.559482,null,31.791702,40.463667,null,51.165691,52.132633,null,55.378051,31.791702,null,40.463667,41.87194,null,51.919438,49.817492,null,35.86166,1.352083,null,35.86166,22.396428,null,40.463667,39.399872,null,40.463667,31.791702,null,14.058324,36.204824,null,38.963745,52.132633,null,63.397768,60.472024,null,35.86166,-40.900557,null,14.058324,51.165691,null,38.963745,41.87194,null,51.165691,50.850346,null,35.86166,39.074208,null,42.315407,51.919438,null,52.132633,40.463667,null,50.850346,46.603354,null,49.817492,51.919438,null,39.399872,49.817492,null,51.919438,52.132633,null,52.132633,41.87194,null,45.943161,51.165691,null,35.86166,4.570868,null,51.165691,63.397768,null,12.879721,36.204824,null,50.850346,52.132633,null,49.817492,37.09024,null,52.132633,37.09024,null,45.943161,51.919438,null,35.86166,46.151241,null,35.86166,31.046051,null,-23.442503,-14.235004,null,33.886917,46.603354,null,1.352083,-25.274398,null,35.86166,56.26392,null,35.86166,26.820553,null,38.963745,47.411631,null,56.26392,63.397768,null,47.411631,45.943161,null,46.603354,40.463667,null,20.593684,9.081999,null,51.165691,55.378051,null,35.86166,8.537981,null,56.26392,60.472024,null,38.963745,46.603354,null,55.169438,63.397768,null,46.603354,41.87194,null,35.86166,-1.831239,null,14.058324,35.907757,null,35.86166,21.916221,null,56.26392,46.603354,null,35.86166,33.223191,null,37.09024,50.850346,null,50.850346,56.26392,null,21.916221,37.09024,null,38.963745,40.463667,null,41.87194,46.603354,null,36.204824,14.058324,null,35.86166,46.818188,null,51.165691,56.26392,null,39.399872,46.603354,null,50.850346,51.165691,null,35.86166,48.019573,null,46.603354,51.165691,null,35.907757,22.396428,null,35.86166,9.081999,null,35.86166,-38.416097,null,35.907757,36.204824,null,35.86166,39.399872,null,56.26392,61.892635,null,38.963745,39.074208,null,30.375321,55.378051,null,63.397768,56.26392,null,35.86166,12.565679,null,51.919438,48.669026,null],"line":{"color":"rgba(0, 150, 0, 0.3)","width":0.6521429603109848},"lon":[104.195397,10.451526,null,5.291266,10.451526,null,104.195397,5.291266,null,104.195397,-3.435973,null,104.195397,133.775136,null,104.195397,-106.346771,null,104.195397,100.992541,null,-95.712891,-106.346771,null,104.195397,-102.552784,null,104.195397,1.888334,null,104.195397,121.774017,null,5.291266,1.888334,null,104.195397,127.766922,null,104.195397,113.921327,null,5.291266,4.351721,null,78.96288,53.847818,null,104.195397,-71.542969,null,19.145136,10.451526,null,104.195397,-3.74922,null,10.451526,1.888334,null,46.869107,1.888334,null,104.195397,101.975766,null,104.195397,19.145136,null,10.451526,19.145136,null,5.291266,19.145136,null,104.195397,12.56738,null,108.277199,-95.712891,null,35.243322,-95.712891,null,104.195397,-51.92528,null,-70.162651,-95.712891,null,104.195397,108.277199,null,104.195397,4.351721,null,10.451526,14.550072,null,104.195397,78.96288,null,69.345116,-95.712891,null,-3.74922,1.888334,null,78.96288,-95.712891,null,104.195397,53.847818,null,104.195397,45.079162,null,104.195397,105.318756,null,100.992541,-95.712891,null,104.195397,16.354896,null,-106.346771,-95.712891,null,10.451526,8.227512,null,10.451526,12.56738,null,104.195397,-75.015152,null,35.243322,10.451526,null,10.451526,15.472962,null,10.451526,-3.74922,null,15.472962,10.451526,null,104.990963,-95.712891,null,9.501785,10.451526,null,-8.224454,-3.74922,null,104.195397,22.937506,null,-7.09262,-3.74922,null,10.451526,5.291266,null,-3.435973,-7.09262,null,-3.74922,12.56738,null,19.145136,15.472962,null,104.195397,103.819836,null,104.195397,114.109497,null,-3.74922,-8.224454,null,-3.74922,-7.09262,null,108.277199,138.252924,null,35.243322,5.291266,null,16.354896,8.468946,null,104.195397,174.885971,null,108.277199,10.451526,null,35.243322,12.56738,null,10.451526,4.351721,null,104.195397,21.824312,null,43.356892,19.145136,null,5.291266,-3.74922,null,4.351721,1.888334,null,15.472962,19.145136,null,-8.224454,15.472962,null,19.145136,5.291266,null,5.291266,12.56738,null,24.96676,10.451526,null,104.195397,-74.297333,null,10.451526,16.354896,null,121.774017,138.252924,null,4.351721,5.291266,null,15.472962,-95.712891,null,5.291266,-95.712891,null,24.96676,19.145136,null,104.195397,14.995463,null,104.195397,34.851612,null,-58.443832,-51.92528,null,9.537499,1.888334,null,103.819836,133.775136,null,104.195397,9.501785,null,104.195397,30.802498,null,35.243322,28.369885,null,9.501785,16.354896,null,28.369885,24.96676,null,1.888334,-3.74922,null,78.96288,8.675277,null,10.451526,-3.435973,null,104.195397,-80.782127,null,9.501785,8.468946,null,35.243322,1.888334,null,23.881275,16.354896,null,1.888334,12.56738,null,104.195397,-78.183406,null,108.277199,127.766922,null,104.195397,95.955974,null,9.501785,1.888334,null,104.195397,43.679291,null,-95.712891,4.351721,null,4.351721,9.501785,null,95.955974,-95.712891,null,35.243322,-3.74922,null,12.56738,1.888334,null,138.252924,108.277199,null,104.195397,8.227512,null,10.451526,9.501785,null,-8.224454,1.888334,null,4.351721,10.451526,null,104.195397,66.923684,null,1.888334,10.451526,null,127.766922,114.109497,null,104.195397,8.675277,null,104.195397,-63.616672,null,127.766922,138.252924,null,104.195397,-8.224454,null,9.501785,-6.911806,null,35.243322,21.824312,null,69.345116,-3.435973,null,16.354896,9.501785,null,104.195397,104.990963,null,19.145136,19.699024,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[-38.416097,-23.442503,-32.522779,40.069099,61.52401,-25.274398,23.424076,35.86166,55.378051,22.396428,36.204824,-40.900557,-6.314993,1.352083,15.870032,37.09024,47.516231,42.733883,46.818188,49.817492,51.165691,46.603354,39.074208,45.1,47.162494,41.87194,51.919438,45.943161,48.669026,46.151241,50.850346,56.26392,28.0,40.463667,61.92411,53.41291,49.815273,56.879635,52.132633,39.399872,63.397768,33.886917,-30.559482,43.915886,56.130366,42.315407,44.016521,26.02751,53.709807,-16.290154,-14.235004,4.570868,-22.328474,-19.015438,23.634501,38.963745,-35.675147,-9.189967,-11.202692,41.0,40.143105,9.30769,12.238333,23.685,17.189877,4.535277,7.539989,7.369722,-2.1646,-0.228021,9.748917,21.521757,35.126413,11.825138,18.735693,-1.831239,26.820553,58.595272,9.145,-17.713371,-0.803689,7.946527,9.945587,15.783471,4.860416,15.199999,18.971187,-0.789275,20.593684,32.427908,33.223191,31.046051,18.109581,30.585164,48.019573,-0.023559,41.20438,12.565679,35.907757,29.31166,19.85627,33.854721,6.428055,26.3351,7.873054,55.169438,22.198745,31.791702,-18.766947,3.202778,17.570692,35.937496,21.916221,46.862496,-18.665695,21.00789,-20.348404,-13.254308,4.210484,-22.95764,-20.904305,9.081999,12.865416,60.472024,28.394857,21.512583,30.375321,8.537981,12.879721,40.339852,25.354826,-1.940278,23.885942,12.862807,14.497401,13.794185,5.152149,3.919305,8.619543,38.861034,38.969719,10.691803,-6.369028,1.373333,48.379433,41.377491,6.42375,14.058324,15.552727,-13.133897,47.411631,64.963051,41.608635,61.892635,71.706936,42.546245,-17.679742,1.650801,33.0,42.708678,34.802075,25.03428,13.193887,12.16957,19.3133,-29.609988,-26.522503],"lon":[-63.616672,-58.443832,-55.765835,45.038189,105.318756,133.775136,53.847818,104.195397,-3.435973,114.109497,138.252924,174.885971,143.95555,103.819836,100.992541,-95.712891,14.550072,25.48583,8.227512,15.472962,10.451526,1.888334,21.824312,15.2,19.503304,12.56738,19.145136,24.96676,19.699024,14.995463,4.351721,9.501785,3.0,-3.74922,25.748151,-8.24389,6.129583,24.603189,5.291266,-8.224454,16.354896,9.537499,22.937506,17.679076,-106.346771,43.356892,21.005859,50.55096,27.953389,-63.588653,-51.92528,-74.297333,24.684866,29.154857,-102.552784,35.243322,-71.542969,-75.015152,17.873887,20.0,47.576927,2.315834,-1.561593,90.3563,-88.49765,114.727669,-5.54708,12.354722,24.15536,15.827659,-83.753428,-77.781167,33.429859,42.590275,-70.162651,-78.183406,30.802498,25.013607,40.489673,178.065033,11.609444,-1.023194,-9.696645,-90.230759,-58.93018,-86.241905,-72.285215,113.921327,78.96288,53.688046,43.679291,34.851612,-77.297508,36.238414,66.923684,37.906193,74.766098,104.990963,127.766922,47.481766,102.495496,35.862285,-9.429499,17.228331,80.771797,23.881275,113.543873,-7.09262,46.869107,73.22068,-3.996166,14.375416,95.955974,103.846656,35.529562,-10.940835,57.552152,34.301525,101.975766,18.49041,165.618042,8.675277,-85.207229,8.468946,84.124008,55.923255,69.345116,-80.782127,121.774017,127.510093,51.183884,29.873888,45.079162,30.217636,-14.452362,-88.89653,46.199616,-56.027783,0.824782,71.276093,59.556278,-61.222503,34.888822,32.290275,31.16558,64.585262,-66.58973,108.277199,48.516388,27.849332,28.369885,-19.020835,21.745275,-6.911806,-42.604303,1.601554,-149.406843,10.267895,65.0,19.37439,38.996815,-77.39628,-59.543198,-68.990021,-81.2546,28.233608,31.465866],"marker":{"color":"red","opacity":0.8,"size":[3.2,2.8,2.8,0.4,6.4,14.0,7.6,20.0,20.0,14.8,15.6,5.6,0.8,11.2,15.6,20.0,12.8,10.8,10.8,20.0,20.0,20.0,10.8,10.4,17.2,20.0,20.0,19.6,17.6,12.8,20.0,20.0,2.4,20.0,8.0,9.6,3.2,7.6,20.0,16.0,20.0,6.8,8.4,3.6,12.0,5.6,8.4,2.0,2.0,1.2,7.2,4.4,1.2,1.2,12.4,20.0,4.8,1.6,2.0,1.2,0.8,0.4,0.8,2.0,0.8,0.8,1.2,0.4,0.8,0.8,1.2,0.4,3.2,0.8,2.4,1.6,1.6,6.4,0.4,0.8,0.4,2.4,0.8,3.2,0.8,2.4,1.2,6.0,20.0,1.2,1.6,6.4,0.8,1.6,2.8,2.0,1.6,8.4,20.0,1.6,2.0,1.6,0.8,1.2,0.8,12.4,0.8,6.4,1.2,0.4,1.6,1.6,4.0,0.8,1.2,0.8,1.2,0.4,14.4,0.8,0.8,1.6,1.6,8.0,0.8,1.2,13.2,1.2,6.4,0.4,2.0,0.4,4.4,1.2,1.6,3.2,0.4,0.4,1.2,0.8,0.8,0.8,1.2,0.4,8.0,3.2,0.8,17.2,0.4,0.8,4.0,2.0,1.6,0.8,0.4,0.4,0.4,0.4,0.8,0.8,0.4,0.4,0.4,0.4,0.4,0.4,0.4]},"mode":"markers","name":"P\u00f3s-Pandemia","text":["ARG\u003cbr\u003eConex\u00f5es: 8","PRY\u003cbr\u003eConex\u00f5es: 7","URY\u003cbr\u003eConex\u00f5es: 7","ARM\u003cbr\u003eConex\u00f5es: 1","RUS\u003cbr\u003eConex\u00f5es: 16","AUS\u003cbr\u003eConex\u00f5es: 35","ARE\u003cbr\u003eConex\u00f5es: 19","CHN\u003cbr\u003eConex\u00f5es: 176","GBR\u003cbr\u003eConex\u00f5es: 72","HKG\u003cbr\u003eConex\u00f5es: 37","JPN\u003cbr\u003eConex\u00f5es: 39","NZL\u003cbr\u003eConex\u00f5es: 14","PNG\u003cbr\u003eConex\u00f5es: 2","SGP\u003cbr\u003eConex\u00f5es: 28","THA\u003cbr\u003eConex\u00f5es: 39","USA\u003cbr\u003eConex\u00f5es: 107","AUT\u003cbr\u003eConex\u00f5es: 32","BGR\u003cbr\u003eConex\u00f5es: 27","CHE\u003cbr\u003eConex\u00f5es: 27","CZE\u003cbr\u003eConex\u00f5es: 58","DEU\u003cbr\u003eConex\u00f5es: 111","FRA\u003cbr\u003eConex\u00f5es: 82","GRC\u003cbr\u003eConex\u00f5es: 27","HRV\u003cbr\u003eConex\u00f5es: 26","HUN\u003cbr\u003eConex\u00f5es: 43","ITA\u003cbr\u003eConex\u00f5es: 67","POL\u003cbr\u003eConex\u00f5es: 72","ROU\u003cbr\u003eConex\u00f5es: 49","SVK\u003cbr\u003eConex\u00f5es: 44","SVN\u003cbr\u003eConex\u00f5es: 32","BEL\u003cbr\u003eConex\u00f5es: 55","DNK\u003cbr\u003eConex\u00f5es: 58","DZA\u003cbr\u003eConex\u00f5es: 6","ESP\u003cbr\u003eConex\u00f5es: 56","FIN\u003cbr\u003eConex\u00f5es: 20","IRL\u003cbr\u003eConex\u00f5es: 24","LUX\u003cbr\u003eConex\u00f5es: 8","LVA\u003cbr\u003eConex\u00f5es: 19","NLD\u003cbr\u003eConex\u00f5es: 77","PRT\u003cbr\u003eConex\u00f5es: 40","SWE\u003cbr\u003eConex\u00f5es: 58","TUN\u003cbr\u003eConex\u00f5es: 17","ZAF\u003cbr\u003eConex\u00f5es: 21","BIH\u003cbr\u003eConex\u00f5es: 9","CAN\u003cbr\u003eConex\u00f5es: 30","GEO\u003cbr\u003eConex\u00f5es: 14","SRB\u003cbr\u003eConex\u00f5es: 21","BHR\u003cbr\u003eConex\u00f5es: 5","BLR\u003cbr\u003eConex\u00f5es: 5","BOL\u003cbr\u003eConex\u00f5es: 3","BRA\u003cbr\u003eConex\u00f5es: 18","COL\u003cbr\u003eConex\u00f5es: 11","BWA\u003cbr\u003eConex\u00f5es: 3","ZWE\u003cbr\u003eConex\u00f5es: 3","MEX\u003cbr\u003eConex\u00f5es: 31","TUR\u003cbr\u003eConex\u00f5es: 95","CHL\u003cbr\u003eConex\u00f5es: 12","PER\u003cbr\u003eConex\u00f5es: 4","AGO\u003cbr\u003eConex\u00f5es: 5","ALB\u003cbr\u003eConex\u00f5es: 3","AZE\u003cbr\u003eConex\u00f5es: 2","BEN\u003cbr\u003eConex\u00f5es: 1","BFA\u003cbr\u003eConex\u00f5es: 2","BGD\u003cbr\u003eConex\u00f5es: 5","BLZ\u003cbr\u003eConex\u00f5es: 2","BRN\u003cbr\u003eConex\u00f5es: 2","CIV\u003cbr\u003eConex\u00f5es: 3","CMR\u003cbr\u003eConex\u00f5es: 1","COD\u003cbr\u003eConex\u00f5es: 2","COG\u003cbr\u003eConex\u00f5es: 2","CRI\u003cbr\u003eConex\u00f5es: 3","CUB\u003cbr\u003eConex\u00f5es: 1","CYP\u003cbr\u003eConex\u00f5es: 8","DJI\u003cbr\u003eConex\u00f5es: 2","DOM\u003cbr\u003eConex\u00f5es: 6","ECU\u003cbr\u003eConex\u00f5es: 4","EGY\u003cbr\u003eConex\u00f5es: 4","EST\u003cbr\u003eConex\u00f5es: 16","ETH\u003cbr\u003eConex\u00f5es: 1","FJI\u003cbr\u003eConex\u00f5es: 2","GAB\u003cbr\u003eConex\u00f5es: 1","GHA\u003cbr\u003eConex\u00f5es: 6","GIN\u003cbr\u003eConex\u00f5es: 2","GTM\u003cbr\u003eConex\u00f5es: 8","GUY\u003cbr\u003eConex\u00f5es: 2","HND\u003cbr\u003eConex\u00f5es: 6","HTI\u003cbr\u003eConex\u00f5es: 3","IDN\u003cbr\u003eConex\u00f5es: 15","IND\u003cbr\u003eConex\u00f5es: 72","IRN\u003cbr\u003eConex\u00f5es: 3","IRQ\u003cbr\u003eConex\u00f5es: 4","ISR\u003cbr\u003eConex\u00f5es: 16","JAM\u003cbr\u003eConex\u00f5es: 2","JOR\u003cbr\u003eConex\u00f5es: 4","KAZ\u003cbr\u003eConex\u00f5es: 7","KEN\u003cbr\u003eConex\u00f5es: 5","KGZ\u003cbr\u003eConex\u00f5es: 4","KHM\u003cbr\u003eConex\u00f5es: 21","KOR\u003cbr\u003eConex\u00f5es: 50","KWT\u003cbr\u003eConex\u00f5es: 4","LAO\u003cbr\u003eConex\u00f5es: 5","LBN\u003cbr\u003eConex\u00f5es: 4","LBR\u003cbr\u003eConex\u00f5es: 2","LBY\u003cbr\u003eConex\u00f5es: 3","LKA\u003cbr\u003eConex\u00f5es: 2","LTU\u003cbr\u003eConex\u00f5es: 31","MAC\u003cbr\u003eConex\u00f5es: 2","MAR\u003cbr\u003eConex\u00f5es: 16","MDG\u003cbr\u003eConex\u00f5es: 3","MDV\u003cbr\u003eConex\u00f5es: 1","MLI\u003cbr\u003eConex\u00f5es: 4","MLT\u003cbr\u003eConex\u00f5es: 4","MMR\u003cbr\u003eConex\u00f5es: 10","MNG\u003cbr\u003eConex\u00f5es: 2","MOZ\u003cbr\u003eConex\u00f5es: 3","MRT\u003cbr\u003eConex\u00f5es: 2","MUS\u003cbr\u003eConex\u00f5es: 3","MWI\u003cbr\u003eConex\u00f5es: 1","MYS\u003cbr\u003eConex\u00f5es: 36","NAM\u003cbr\u003eConex\u00f5es: 2","NCL\u003cbr\u003eConex\u00f5es: 2","NGA\u003cbr\u003eConex\u00f5es: 4","NIC\u003cbr\u003eConex\u00f5es: 4","NOR\u003cbr\u003eConex\u00f5es: 20","NPL\u003cbr\u003eConex\u00f5es: 2","OMN\u003cbr\u003eConex\u00f5es: 3","PAK\u003cbr\u003eConex\u00f5es: 33","PAN\u003cbr\u003eConex\u00f5es: 3","PHL\u003cbr\u003eConex\u00f5es: 16","PRK\u003cbr\u003eConex\u00f5es: 1","QAT\u003cbr\u003eConex\u00f5es: 5","RWA\u003cbr\u003eConex\u00f5es: 1","SAU\u003cbr\u003eConex\u00f5es: 11","SDN\u003cbr\u003eConex\u00f5es: 3","SEN\u003cbr\u003eConex\u00f5es: 4","SLV\u003cbr\u003eConex\u00f5es: 8","SOM\u003cbr\u003eConex\u00f5es: 1","SUR\u003cbr\u003eConex\u00f5es: 1","TGO\u003cbr\u003eConex\u00f5es: 3","TJK\u003cbr\u003eConex\u00f5es: 2","TKM\u003cbr\u003eConex\u00f5es: 2","TTO\u003cbr\u003eConex\u00f5es: 2","TZA\u003cbr\u003eConex\u00f5es: 3","UGA\u003cbr\u003eConex\u00f5es: 1","UKR\u003cbr\u003eConex\u00f5es: 20","UZB\u003cbr\u003eConex\u00f5es: 8","VEN\u003cbr\u003eConex\u00f5es: 2","VNM\u003cbr\u003eConex\u00f5es: 43","YEM\u003cbr\u003eConex\u00f5es: 1","ZMB\u003cbr\u003eConex\u00f5es: 2","MDA\u003cbr\u003eConex\u00f5es: 10","ISL\u003cbr\u003eConex\u00f5es: 5","MKD\u003cbr\u003eConex\u00f5es: 4","FRO\u003cbr\u003eConex\u00f5es: 2","GRL\u003cbr\u003eConex\u00f5es: 1","AND\u003cbr\u003eConex\u00f5es: 1","PYF\u003cbr\u003eConex\u00f5es: 1","GNQ\u003cbr\u003eConex\u00f5es: 1","AFG\u003cbr\u003eConex\u00f5es: 2","MNE\u003cbr\u003eConex\u00f5es: 2","SYR\u003cbr\u003eConex\u00f5es: 1","BHS\u003cbr\u003eConex\u00f5es: 1","BRB\u003cbr\u003eConex\u00f5es: 1","CUW\u003cbr\u003eConex\u00f5es: 1","CYM\u003cbr\u003eConex\u00f5es: 1","LSO\u003cbr\u003eConex\u00f5es: 1","SWZ\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"projection":{"type":"equirectangular"},"showland":true,"landcolor":"lightgray","showocean":true,"oceancolor":"lightblue","showcountries":true,"countrycolor":"white"},"title":{"text":"Evolu\u00e7\u00e3o das Redes de Exporta\u00e7\u00e3o de M\u00e1scaras Cir\u00fargicas"},"height":600,"sliders":[{"active":0,"currentvalue":{"prefix":"Per\u00edodo: "},"steps":[{"args":[{"visible":[true,true,true,true,false,false,false,false,false,false,false,false]}],"label":"Pr\u00e9-Pandemia","method":"update"},{"args":[{"visible":[false,false,false,false,true,true,true,true,false,false,false,false]}],"label":"Durante Pandemia","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,true,true,true,true]}],"label":"P\u00f3s-Pandemia","method":"update"}]}]},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('611de89c-b139-42b2-ad77-9cdd65cc4f30');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
df.columns
```




    Index(['refYear', 'refMonth', 'flowDesc', 'reporterISO', 'partnerISO', 'qty',
           'primaryValue', 'unitPrice', 'Exportador', 'Importador'],
          dtype='object')




```python
def create_network(df, year, threshold=0):

    G = nx.DiGraph(name=f"Rede de Exportação de Máscaras Cirúrgicas - {year}")
    
    for _, row in df.iterrows():
        if row['qty'] > threshold:
            G.add_edge(
                row['reporterISO'],
                row['partnerISO'],
                weight=row['qty']
            )
    
    return G

networks_anos = {}
for ano in range(2015, 2025):  
    # filtrar os dados para um ano específico
    dados_ano = df_exports[df_exports['refYear'] == ano]
    networks_anos[str(ano)] = create_network(dados_ano, str(ano), threshold=0)

fig = criar_mapa_temp(networks_anos, coords)
fig.show()
fig.write_html("evolucao_redes_mascaras_anual.html")
```


<div>                            <div id="cfe26ae9-70d6-44d7-bf3d-e77eb6d8b1c2" class="plotly-graph-div" style="height:600px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("cfe26ae9-70d6-44d7-bf3d-e77eb6d8b1c2")) {                    Plotly.newPlot(                        "cfe26ae9-70d6-44d7-bf3d-e77eb6d8b1c2",                        [{"hoverinfo":"none","lat":[22.396428,37.09024,null,22.396428,35.86166,null,23.634501,37.09024,null,14.058324,37.09024,null,22.396428,36.204824,null,22.396428,51.165691,null,22.396428,55.378051,null,52.132633,51.165691,null,22.396428,-25.274398,null,31.791702,40.463667,null,22.396428,46.603354,null,22.396428,-0.789275,null,22.396428,51.919438,null,22.396428,52.132633,null,51.165691,51.919438,null,22.396428,56.130366,null,18.735693,37.09024,null,51.165691,46.603354,null,51.165691,47.516231,null,51.165691,41.87194,null,22.396428,-0.228021,null,51.165691,52.132633,null,22.396428,41.87194,null,52.132633,50.850346,null,22.396428,50.850346,null,22.396428,20.593684,null,14.058324,51.165691,null,22.396428,7.946527,null,22.396428,22.198745,null,22.396428,-14.235004,null,22.396428,14.058324,null,51.165691,46.818188,null,14.058324,36.204824,null,52.132633,46.603354,null,51.165691,63.397768,null,15.870032,51.165691,null,22.396428,23.424076,null,51.165691,40.463667,null,22.396428,56.26392,null,38.963745,37.09024,null,51.165691,49.817492,null,22.396428,4.210484,null,22.396428,40.463667,null,38.963745,51.165691,null,51.165691,56.26392,null,22.396428,6.42375,null,52.132633,40.463667,null,63.397768,60.472024,null,51.165691,55.378051,null,15.870032,37.09024,null,22.396428,23.634501,null,22.396428,15.783471,null,15.870032,49.817492,null,22.396428,1.352083,null,14.058324,55.378051,null,35.907757,14.058324,null,52.132633,41.87194,null,51.165691,37.09024,null,52.132633,55.378051,null,63.397768,61.92411,null,50.850346,46.603354,null,40.463667,39.399872,null,38.963745,52.132633,null,51.165691,50.850346,null,14.058324,41.87194,null,22.396428,44.016521,null,22.396428,47.516231,null,-0.789275,36.204824,null,22.396428,41.608635,null,63.397768,56.26392,null,35.907757,35.86166,null,52.132633,51.919438,null,38.963745,47.411631,null,51.165691,38.963745,null,14.058324,46.603354,null,14.058324,52.132633,null,22.396428,12.879721,null,38.963745,33.223191,null,51.165691,23.424076,null,22.396428,63.397768,null,14.058324,35.907757,null,52.132633,37.09024,null,40.463667,31.791702,null,22.396428,-40.900557,null,22.396428,31.046051,null,40.463667,46.603354,null,14.058324,40.463667,null,15.870032,36.204824,null,51.165691,35.86166,null,14.058324,51.919438,null,50.850346,52.132633,null,51.165691,61.92411,null,14.058324,47.516231,null,50.850346,51.165691,null,51.165691,61.52401,null,15.870032,-25.274398,null,22.396428,60.472024,null,22.396428,23.685,null,14.058324,63.397768,null,22.396428,9.081999,null,40.463667,51.165691,null,23.885942,29.31166,null,22.396428,49.817492,null,38.963745,63.397768,null,35.907757,1.352083,null,63.397768,55.378051,null,22.396428,35.907757,null,40.463667,41.87194,null,35.907757,36.204824,null,35.907757,22.396428,null,22.396428,-30.559482,null,51.165691,47.162494,null,38.963745,48.669026,null,22.396428,38.963745,null,56.879635,63.397768,null,51.165691,45.943161,null,22.396428,13.444304,null,22.396428,33.886917,null,14.058324,35.86166,null,51.165691,48.669026,null,22.396428,58.595272,null,-22.328474,-19.015438,null,23.634501,15.783471,null,52.132633,47.516231,null,15.870032,-0.789275,null,38.963745,46.603354,null,22.396428,12.565679,null,63.397768,51.165691,null,53.41291,36.204824,null,15.870032,52.132633,null,35.907757,-0.789275,null,50.850346,40.463667,null,-30.559482,55.378051,null,38.963745,55.378051,null,51.165691,-0.789275,null,38.963745,23.885942,null,38.963745,39.074208,null,22.396428,46.818188,null,51.165691,60.472024,null,35.907757,12.565679,null,15.870032,50.850346,null,22.396428,61.92411,null,14.058324,61.52401,null,14.058324,56.879635,null,31.791702,46.603354,null,52.132633,63.397768,null,22.396428,46.151241,null,51.165691,33.886917,null,48.669026,51.165691,null,51.165691,-30.559482,null,23.885942,23.424076,null,-30.559482,-19.015438,null,35.907757,37.09024,null,38.963745,41.87194,null,14.058324,56.130366,null,52.132633,49.817492,null,22.396428,7.873054,null,22.396428,-0.023559,null,51.165691,46.151241,null,14.058324,-25.274398,null,38.963745,31.791702,null,51.165691,53.41291,null,52.132633,46.818188,null,22.396428,7.369722,null,22.396428,47.162494,null,51.165691,26.820553,null,48.669026,46.603354,null,51.165691,49.815273,null,51.165691,39.074208,null,38.963745,51.919438,null,53.41291,51.165691,null,63.397768,51.919438,null,13.794185,37.09024,null,51.165691,45.1,null,22.396428,-38.416097,null,51.165691,42.733883,null,40.463667,-20.348404,null,52.132633,61.52401,null,51.165691,23.885942,null,38.963745,32.427908,null,53.41291,55.378051,null,35.907757,-40.900557,null,63.397768,50.850346,null,51.165691,39.399872,null,14.058324,6.42375,null,22.396428,-29.609988,null,22.396428,25.354826,null,15.870032,4.210484,null,46.818188,51.165691,null,50.850346,55.378051,null,22.396428,45.1,null,14.058324,23.424076,null,35.907757,23.424076,null,15.870032,63.397768,null,51.165691,55.169438,null,52.132633,38.963745,null,51.165691,44.016521,null,52.132633,56.26392,null,15.870032,-35.675147,null,63.397768,37.09024,null,63.397768,-25.274398,null,52.132633,61.92411,null,41.608635,52.132633,null,22.396428,18.109581,null,22.396428,-19.015438,null,15.870032,23.885942,null,52.132633,47.162494,null,22.396428,26.02751,null,22.396428,35.937496,null,52.132633,60.472024,null,38.963745,31.046051,null,22.396428,31.791702,null,52.132633,48.669026,null,15.870032,35.86166,null,-9.189967,-1.831239,null,-30.559482,-11.202692,null,-30.559482,-2.1646,null,-30.559482,-22.328474,null,38.963745,30.585164,null,38.963745,40.463667,null,50.850346,53.41291,null,50.850346,51.919438,null,52.132633,23.885942,null,48.669026,47.516231,null,51.165691,-25.274398,null,14.058324,41.377491,null,51.165691,31.046051,null,38.963745,42.733883,null,38.963745,38.969719,null,-0.789275,-25.274398,null,48.669026,47.162494,null,51.165691,-14.235004,null,40.463667,52.132633,null,23.634501,15.199999,null,22.396428,42.733883,null,22.396428,61.52401,null,-30.559482,-22.95764,null,63.397768,46.603354,null,-18.766947,46.603354,null,22.396428,-21.178986,null,15.870032,7.946527,null,22.396428,48.379433,null,-0.789275,23.885942,null,38.963745,56.26392,null,51.165691,20.593684,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.5015285927455511},"lon":[114.109497,-95.712891,null,114.109497,104.195397,null,-102.552784,-95.712891,null,108.277199,-95.712891,null,114.109497,138.252924,null,114.109497,10.451526,null,114.109497,-3.435973,null,5.291266,10.451526,null,114.109497,133.775136,null,-7.09262,-3.74922,null,114.109497,1.888334,null,114.109497,113.921327,null,114.109497,19.145136,null,114.109497,5.291266,null,10.451526,19.145136,null,114.109497,-106.346771,null,-70.162651,-95.712891,null,10.451526,1.888334,null,10.451526,14.550072,null,10.451526,12.56738,null,114.109497,15.827659,null,10.451526,5.291266,null,114.109497,12.56738,null,5.291266,4.351721,null,114.109497,4.351721,null,114.109497,78.96288,null,108.277199,10.451526,null,114.109497,-1.023194,null,114.109497,113.543873,null,114.109497,-51.92528,null,114.109497,108.277199,null,10.451526,8.227512,null,108.277199,138.252924,null,5.291266,1.888334,null,10.451526,16.354896,null,100.992541,10.451526,null,114.109497,53.847818,null,10.451526,-3.74922,null,114.109497,9.501785,null,35.243322,-95.712891,null,10.451526,15.472962,null,114.109497,101.975766,null,114.109497,-3.74922,null,35.243322,10.451526,null,10.451526,9.501785,null,114.109497,-66.58973,null,5.291266,-3.74922,null,16.354896,8.468946,null,10.451526,-3.435973,null,100.992541,-95.712891,null,114.109497,-102.552784,null,114.109497,-90.230759,null,100.992541,15.472962,null,114.109497,103.819836,null,108.277199,-3.435973,null,127.766922,108.277199,null,5.291266,12.56738,null,10.451526,-95.712891,null,5.291266,-3.435973,null,16.354896,25.748151,null,4.351721,1.888334,null,-3.74922,-8.224454,null,35.243322,5.291266,null,10.451526,4.351721,null,108.277199,12.56738,null,114.109497,21.005859,null,114.109497,14.550072,null,113.921327,138.252924,null,114.109497,21.745275,null,16.354896,9.501785,null,127.766922,104.195397,null,5.291266,19.145136,null,35.243322,28.369885,null,10.451526,35.243322,null,108.277199,1.888334,null,108.277199,5.291266,null,114.109497,121.774017,null,35.243322,43.679291,null,10.451526,53.847818,null,114.109497,16.354896,null,108.277199,127.766922,null,5.291266,-95.712891,null,-3.74922,-7.09262,null,114.109497,174.885971,null,114.109497,34.851612,null,-3.74922,1.888334,null,108.277199,-3.74922,null,100.992541,138.252924,null,10.451526,104.195397,null,108.277199,19.145136,null,4.351721,5.291266,null,10.451526,25.748151,null,108.277199,14.550072,null,4.351721,10.451526,null,10.451526,105.318756,null,100.992541,133.775136,null,114.109497,8.468946,null,114.109497,90.3563,null,108.277199,16.354896,null,114.109497,8.675277,null,-3.74922,10.451526,null,45.079162,47.481766,null,114.109497,15.472962,null,35.243322,16.354896,null,127.766922,103.819836,null,16.354896,-3.435973,null,114.109497,127.766922,null,-3.74922,12.56738,null,127.766922,138.252924,null,127.766922,114.109497,null,114.109497,22.937506,null,10.451526,19.503304,null,35.243322,19.699024,null,114.109497,35.243322,null,24.603189,16.354896,null,10.451526,24.96676,null,114.109497,144.793731,null,114.109497,9.537499,null,108.277199,104.195397,null,10.451526,19.699024,null,114.109497,25.013607,null,24.684866,29.154857,null,-102.552784,-90.230759,null,5.291266,14.550072,null,100.992541,113.921327,null,35.243322,1.888334,null,114.109497,104.990963,null,16.354896,10.451526,null,-8.24389,138.252924,null,100.992541,5.291266,null,127.766922,113.921327,null,4.351721,-3.74922,null,22.937506,-3.435973,null,35.243322,-3.435973,null,10.451526,113.921327,null,35.243322,45.079162,null,35.243322,21.824312,null,114.109497,8.227512,null,10.451526,8.468946,null,127.766922,104.990963,null,100.992541,4.351721,null,114.109497,25.748151,null,108.277199,105.318756,null,108.277199,24.603189,null,-7.09262,1.888334,null,5.291266,16.354896,null,114.109497,14.995463,null,10.451526,9.537499,null,19.699024,10.451526,null,10.451526,22.937506,null,45.079162,53.847818,null,22.937506,29.154857,null,127.766922,-95.712891,null,35.243322,12.56738,null,108.277199,-106.346771,null,5.291266,15.472962,null,114.109497,80.771797,null,114.109497,37.906193,null,10.451526,14.995463,null,108.277199,133.775136,null,35.243322,-7.09262,null,10.451526,-8.24389,null,5.291266,8.227512,null,114.109497,12.354722,null,114.109497,19.503304,null,10.451526,30.802498,null,19.699024,1.888334,null,10.451526,6.129583,null,10.451526,21.824312,null,35.243322,19.145136,null,-8.24389,10.451526,null,16.354896,19.145136,null,-88.89653,-95.712891,null,10.451526,15.2,null,114.109497,-63.616672,null,10.451526,25.48583,null,-3.74922,57.552152,null,5.291266,105.318756,null,10.451526,45.079162,null,35.243322,53.688046,null,-8.24389,-3.435973,null,127.766922,174.885971,null,16.354896,4.351721,null,10.451526,-8.224454,null,108.277199,-66.58973,null,114.109497,28.233608,null,114.109497,51.183884,null,100.992541,101.975766,null,8.227512,10.451526,null,4.351721,-3.435973,null,114.109497,15.2,null,108.277199,53.847818,null,127.766922,53.847818,null,100.992541,16.354896,null,10.451526,23.881275,null,5.291266,35.243322,null,10.451526,21.005859,null,5.291266,9.501785,null,100.992541,-71.542969,null,16.354896,-95.712891,null,16.354896,133.775136,null,5.291266,25.748151,null,21.745275,5.291266,null,114.109497,-77.297508,null,114.109497,29.154857,null,100.992541,45.079162,null,5.291266,19.503304,null,114.109497,50.55096,null,114.109497,14.375416,null,5.291266,8.468946,null,35.243322,34.851612,null,114.109497,-7.09262,null,5.291266,19.699024,null,100.992541,104.195397,null,-75.015152,-78.183406,null,22.937506,17.873887,null,22.937506,24.15536,null,22.937506,24.684866,null,35.243322,36.238414,null,35.243322,-3.74922,null,4.351721,-8.24389,null,4.351721,19.145136,null,5.291266,45.079162,null,19.699024,14.550072,null,10.451526,133.775136,null,108.277199,64.585262,null,10.451526,34.851612,null,35.243322,25.48583,null,35.243322,59.556278,null,113.921327,133.775136,null,19.699024,19.503304,null,10.451526,-51.92528,null,-3.74922,5.291266,null,-102.552784,-86.241905,null,114.109497,25.48583,null,114.109497,105.318756,null,22.937506,18.49041,null,16.354896,1.888334,null,46.869107,1.888334,null,114.109497,-175.198242,null,100.992541,-1.023194,null,114.109497,31.16558,null,113.921327,45.079162,null,35.243322,9.501785,null,10.451526,78.96288,null],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":true,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,40.463667,-11.202692,50.850346,-14.235004,-22.95764,-6.369028,26.02751,29.31166,23.885942,23.424076,28.0,-25.274398,47.516231,43.915886,42.733883,56.130366,-35.675147,35.86166,-2.1646,45.1,35.126413,49.817492,9.30769,56.26392,1.650801,15.179384,58.595272,61.92411,46.603354,-17.679742,-0.803689,51.165691,36.140751,39.074208,9.945587,22.396428,47.162494,64.963051,33.223191,53.41291,31.046051,41.87194,36.204824,48.019573,30.585164,-0.023559,35.907757,33.854721,56.879635,55.169438,49.815273,4.210484,35.937496,-20.348404,23.634501,31.791702,52.132633,60.472024,-23.442503,-9.189967,51.919438,39.399872,25.354826,45.943161,61.52401,44.016521,8.460555,20.593684,1.352083,48.669026,14.058324,46.151241,-30.559482,63.397768,46.818188,15.870032,38.963745,48.379433,41.608635,55.378051,37.09024,-32.522779,-16.290154,-38.416097,4.570868,9.748917,-1.831239,7.539989,18.109581,-18.665695,8.537981,3.919305,26.820553,4.535277,12.565679,41.0,23.685,32.3078,21.521757,12.16957,33.886917,-0.228021,9.081999,18.735693,18.971187,13.794185,15.199999,12.865416,-17.713371,-15.376706,-40.900557,-6.314993,42.315407,40.143105,40.069099,33.0,53.709807,7.369722,7.873054,61.892635,7.946527,71.706936,15.783471,4.860416,-0.789275,32.427908,41.20438,26.3351,3.202778,46.862496,47.411631,42.708678,21.512583,12.52111,-20.904305,30.375321,12.879721,43.94236,14.497401,-4.679574,12.862807,8.619543,38.969719,1.373333,12.238333,6.42375,15.552727,22.198745,21.916221,19.85627,17.189877,-18.766947,-13.254308,-19.015438,16.742498,17.060816,17.570692,-22.328474,18.420695,-29.609988,21.00789,-26.522503,-13.133897,28.394857,38.861034,27.514162,10.691803,12.262776,12.984305,9.145,7.131474,5.152149,34.802075,41.377491,15.454166,-3.370417,-13.759029,25.03428,13.193887,11.825138,18.04248,12.20189,13.909444,-1.940278,6.611111,13.443182,31.952162,6.428055,17.607789,-21.178986,19.3133,46.946947,15.414999,17.357822,18.218785,13.444304,-8.874217,7.51498,-3.373056,17.897476,21.694025,7.862684,-14.28522,-10.447525,-11.6455,11.803749,-0.522778,-9.64571,-21.236736,-7.109535,16.5388,-15.965,0.18636,7.425554,-90.0,-29.040835],"lon":[1.601554,-3.74922,17.873887,4.351721,-51.92528,18.49041,34.888822,50.55096,47.481766,45.079162,53.847818,3.0,133.775136,14.550072,17.679076,25.48583,-106.346771,-71.542969,104.195397,24.15536,15.2,33.429859,15.472962,2.315834,9.501785,10.267895,39.782334,25.013607,25.748151,1.888334,-149.406843,11.609444,10.451526,-5.353585,21.824312,-9.696645,114.109497,19.503304,-19.020835,43.679291,-8.24389,34.851612,12.56738,138.252924,66.923684,36.238414,37.906193,127.766922,35.862285,24.603189,23.881275,6.129583,101.975766,14.375416,57.552152,-102.552784,-7.09262,5.291266,8.468946,-58.443832,-75.015152,19.145136,-8.224454,51.183884,24.96676,105.318756,21.005859,-11.779889,78.96288,103.819836,19.699024,108.277199,14.995463,22.937506,16.354896,8.227512,100.992541,35.243322,31.16558,21.745275,-3.435973,-95.712891,-55.765835,-63.588653,-63.616672,-74.297333,-83.753428,-78.183406,-5.54708,-77.297508,35.529562,-80.782127,-56.027783,30.802498,114.727669,104.990963,20.0,90.3563,-64.7505,-77.781167,-68.990021,9.537499,15.827659,8.675277,-70.162651,-72.285215,-88.89653,-86.241905,-85.207229,178.065033,166.959158,174.885971,143.95555,43.356892,47.576927,45.038189,65.0,27.953389,12.354722,80.771797,-6.911806,-1.023194,-42.604303,-90.230759,-58.93018,113.921327,53.688046,74.766098,17.228331,73.22068,103.846656,28.369885,19.37439,55.923255,-69.968338,165.618042,69.345116,121.774017,12.457777,-14.452362,55.491977,30.217636,0.824782,59.556278,32.290275,-1.561593,-66.58973,48.516388,113.543873,95.955974,102.495496,-88.49765,46.869107,34.301525,29.154857,-62.187366,-61.796428,-3.996166,24.684866,-64.639968,28.233608,-10.940835,31.465866,27.849332,84.124008,71.276093,90.433601,-61.222503,-61.604171,-61.287228,40.489673,171.184478,46.199616,38.996815,64.585262,18.732207,-168.734039,-172.104629,-77.39628,-59.543198,42.590275,-63.05483,-68.262383,-60.978893,29.873888,20.939444,-15.310139,35.233154,-9.429499,8.081666,-175.198242,-81.2546,-56.32509,-61.370976,-62.782998,-63.043653,144.793731,125.727,134.58252,29.918886,-62.83055,-71.797928,30.217636,-170.70444,105.690449,43.3333,-15.180413,166.931503,160.156194,-159.777671,179.194167,-23.0418,-5.7089,6.613081,150.550812,0.0,167.954712],"marker":{"color":"red","opacity":0.8,"size":[2.4,20.0,13.6,20.0,20.0,4.8,6.4,12.0,12.8,9.2,10.0,5.2,9.2,8.0,4.4,5.6,20.0,17.6,10.4,4.4,6.0,7.6,8.4,2.8,8.4,3.6,1.2,6.4,6.8,12.0,2.4,3.6,20.0,2.8,8.4,3.2,20.0,6.8,4.8,4.8,20.0,7.6,10.4,10.0,5.2,5.6,6.4,20.0,6.4,20.0,6.4,5.6,6.8,4.4,4.8,15.6,20.0,20.0,20.0,8.0,20.0,8.0,7.6,6.4,8.8,9.2,5.2,2.4,7.2,8.0,20.0,20.0,5.2,20.0,20.0,20.0,20.0,20.0,6.8,14.4,14.4,15.2,6.8,8.0,6.0,6.4,6.0,7.6,8.8,5.2,3.6,8.0,3.2,8.0,4.4,5.6,4.8,4.4,2.0,4.4,2.8,6.0,6.4,6.8,15.6,3.2,10.8,5.2,5.2,10.8,2.0,8.8,4.0,13.2,4.8,4.4,2.4,4.4,4.0,4.8,1.2,6.4,2.0,5.2,3.2,20.0,4.8,2.4,2.8,3.6,3.6,4.8,3.2,6.8,3.2,3.2,4.8,7.2,1.6,4.8,3.6,2.8,4.0,3.6,4.0,3.6,4.8,1.6,6.8,3.2,2.4,2.0,9.6,1.6,7.2,0.8,3.6,4.0,4.0,2.8,2.0,3.6,4.4,4.8,2.4,3.2,0.8,14.8,4.0,2.4,6.8,2.0,2.0,1.6,3.2,2.0,0.4,1.6,2.8,2.8,2.8,0.8,0.8,2.8,3.2,2.0,1.2,2.0,4.0,1.6,1.2,3.2,0.4,1.2,1.2,0.8,2.0,0.8,0.4,0.8,0.8,1.6,0.8,0.4,0.4,0.4,0.8,0.4,0.8,0.4,0.4,1.6,0.8,0.8,0.4,0.4,0.4]},"mode":"markers","name":"2015","text":["AND\u003cbr\u003eConex\u00f5es: 6","ESP\u003cbr\u003eConex\u00f5es: 169","AGO\u003cbr\u003eConex\u00f5es: 34","BEL\u003cbr\u003eConex\u00f5es: 172","BRA\u003cbr\u003eConex\u00f5es: 96","NAM\u003cbr\u003eConex\u00f5es: 12","TZA\u003cbr\u003eConex\u00f5es: 16","BHR\u003cbr\u003eConex\u00f5es: 30","KWT\u003cbr\u003eConex\u00f5es: 32","SAU\u003cbr\u003eConex\u00f5es: 23","ARE\u003cbr\u003eConex\u00f5es: 25","DZA\u003cbr\u003eConex\u00f5es: 13","AUS\u003cbr\u003eConex\u00f5es: 23","AUT\u003cbr\u003eConex\u00f5es: 20","BIH\u003cbr\u003eConex\u00f5es: 11","BGR\u003cbr\u003eConex\u00f5es: 14","CAN\u003cbr\u003eConex\u00f5es: 94","CHL\u003cbr\u003eConex\u00f5es: 44","CHN\u003cbr\u003eConex\u00f5es: 26","COD\u003cbr\u003eConex\u00f5es: 11","HRV\u003cbr\u003eConex\u00f5es: 15","CYP\u003cbr\u003eConex\u00f5es: 19","CZE\u003cbr\u003eConex\u00f5es: 21","BEN\u003cbr\u003eConex\u00f5es: 7","DNK\u003cbr\u003eConex\u00f5es: 21","GNQ\u003cbr\u003eConex\u00f5es: 9","ERI\u003cbr\u003eConex\u00f5es: 3","EST\u003cbr\u003eConex\u00f5es: 16","FIN\u003cbr\u003eConex\u00f5es: 17","FRA\u003cbr\u003eConex\u00f5es: 30","PYF\u003cbr\u003eConex\u00f5es: 6","GAB\u003cbr\u003eConex\u00f5es: 9","DEU\u003cbr\u003eConex\u00f5es: 207","GIB\u003cbr\u003eConex\u00f5es: 7","GRC\u003cbr\u003eConex\u00f5es: 21","GIN\u003cbr\u003eConex\u00f5es: 8","HKG\u003cbr\u003eConex\u00f5es: 145","HUN\u003cbr\u003eConex\u00f5es: 17","ISL\u003cbr\u003eConex\u00f5es: 12","IRQ\u003cbr\u003eConex\u00f5es: 12","IRL\u003cbr\u003eConex\u00f5es: 68","ISR\u003cbr\u003eConex\u00f5es: 19","ITA\u003cbr\u003eConex\u00f5es: 26","JPN\u003cbr\u003eConex\u00f5es: 25","KAZ\u003cbr\u003eConex\u00f5es: 13","JOR\u003cbr\u003eConex\u00f5es: 14","KEN\u003cbr\u003eConex\u00f5es: 16","KOR\u003cbr\u003eConex\u00f5es: 144","LBN\u003cbr\u003eConex\u00f5es: 16","LVA\u003cbr\u003eConex\u00f5es: 60","LTU\u003cbr\u003eConex\u00f5es: 16","LUX\u003cbr\u003eConex\u00f5es: 14","MYS\u003cbr\u003eConex\u00f5es: 17","MLT\u003cbr\u003eConex\u00f5es: 11","MUS\u003cbr\u003eConex\u00f5es: 12","MEX\u003cbr\u003eConex\u00f5es: 39","MAR\u003cbr\u003eConex\u00f5es: 86","NLD\u003cbr\u003eConex\u00f5es: 200","NOR\u003cbr\u003eConex\u00f5es: 76","PRY\u003cbr\u003eConex\u00f5es: 20","PER\u003cbr\u003eConex\u00f5es: 70","POL\u003cbr\u003eConex\u00f5es: 20","PRT\u003cbr\u003eConex\u00f5es: 19","QAT\u003cbr\u003eConex\u00f5es: 16","ROU\u003cbr\u003eConex\u00f5es: 22","RUS\u003cbr\u003eConex\u00f5es: 23","SRB\u003cbr\u003eConex\u00f5es: 13","SLE\u003cbr\u003eConex\u00f5es: 6","IND\u003cbr\u003eConex\u00f5es: 18","SGP\u003cbr\u003eConex\u00f5es: 20","SVK\u003cbr\u003eConex\u00f5es: 81","VNM\u003cbr\u003eConex\u00f5es: 85","SVN\u003cbr\u003eConex\u00f5es: 13","ZAF\u003cbr\u003eConex\u00f5es: 127","SWE\u003cbr\u003eConex\u00f5es: 139","CHE\u003cbr\u003eConex\u00f5es: 156","THA\u003cbr\u003eConex\u00f5es: 136","TUR\u003cbr\u003eConex\u00f5es: 160","UKR\u003cbr\u003eConex\u00f5es: 17","MKD\u003cbr\u003eConex\u00f5es: 36","GBR\u003cbr\u003eConex\u00f5es: 36","USA\u003cbr\u003eConex\u00f5es: 38","URY\u003cbr\u003eConex\u00f5es: 17","BOL\u003cbr\u003eConex\u00f5es: 20","ARG\u003cbr\u003eConex\u00f5es: 15","COL\u003cbr\u003eConex\u00f5es: 16","CRI\u003cbr\u003eConex\u00f5es: 15","ECU\u003cbr\u003eConex\u00f5es: 19","CIV\u003cbr\u003eConex\u00f5es: 22","JAM\u003cbr\u003eConex\u00f5es: 13","MOZ\u003cbr\u003eConex\u00f5es: 9","PAN\u003cbr\u003eConex\u00f5es: 20","SUR\u003cbr\u003eConex\u00f5es: 8","EGY\u003cbr\u003eConex\u00f5es: 20","BRN\u003cbr\u003eConex\u00f5es: 11","KHM\u003cbr\u003eConex\u00f5es: 14","ALB\u003cbr\u003eConex\u00f5es: 12","BGD\u003cbr\u003eConex\u00f5es: 11","BMU\u003cbr\u003eConex\u00f5es: 5","CUB\u003cbr\u003eConex\u00f5es: 11","CUW\u003cbr\u003eConex\u00f5es: 7","TUN\u003cbr\u003eConex\u00f5es: 15","COG\u003cbr\u003eConex\u00f5es: 16","NGA\u003cbr\u003eConex\u00f5es: 17","DOM\u003cbr\u003eConex\u00f5es: 39","HTI\u003cbr\u003eConex\u00f5es: 8","SLV\u003cbr\u003eConex\u00f5es: 27","HND\u003cbr\u003eConex\u00f5es: 13","NIC\u003cbr\u003eConex\u00f5es: 13","FJI\u003cbr\u003eConex\u00f5es: 27","VUT\u003cbr\u003eConex\u00f5es: 5","NZL\u003cbr\u003eConex\u00f5es: 22","PNG\u003cbr\u003eConex\u00f5es: 10","GEO\u003cbr\u003eConex\u00f5es: 33","AZE\u003cbr\u003eConex\u00f5es: 12","ARM\u003cbr\u003eConex\u00f5es: 11","AFG\u003cbr\u003eConex\u00f5es: 6","BLR\u003cbr\u003eConex\u00f5es: 11","CMR\u003cbr\u003eConex\u00f5es: 10","LKA\u003cbr\u003eConex\u00f5es: 12","FRO\u003cbr\u003eConex\u00f5es: 3","GHA\u003cbr\u003eConex\u00f5es: 16","GRL\u003cbr\u003eConex\u00f5es: 5","GTM\u003cbr\u003eConex\u00f5es: 13","GUY\u003cbr\u003eConex\u00f5es: 8","IDN\u003cbr\u003eConex\u00f5es: 72","IRN\u003cbr\u003eConex\u00f5es: 12","KGZ\u003cbr\u003eConex\u00f5es: 6","LBY\u003cbr\u003eConex\u00f5es: 7","MDV\u003cbr\u003eConex\u00f5es: 9","MNG\u003cbr\u003eConex\u00f5es: 9","MDA\u003cbr\u003eConex\u00f5es: 12","MNE\u003cbr\u003eConex\u00f5es: 8","OMN\u003cbr\u003eConex\u00f5es: 17","ABW\u003cbr\u003eConex\u00f5es: 8","NCL\u003cbr\u003eConex\u00f5es: 8","PAK\u003cbr\u003eConex\u00f5es: 12","PHL\u003cbr\u003eConex\u00f5es: 18","SMR\u003cbr\u003eConex\u00f5es: 4","SEN\u003cbr\u003eConex\u00f5es: 12","SYC\u003cbr\u003eConex\u00f5es: 9","SDN\u003cbr\u003eConex\u00f5es: 7","TGO\u003cbr\u003eConex\u00f5es: 10","TKM\u003cbr\u003eConex\u00f5es: 9","UGA\u003cbr\u003eConex\u00f5es: 10","BFA\u003cbr\u003eConex\u00f5es: 9","VEN\u003cbr\u003eConex\u00f5es: 12","YEM\u003cbr\u003eConex\u00f5es: 4","MAC\u003cbr\u003eConex\u00f5es: 17","MMR\u003cbr\u003eConex\u00f5es: 8","LAO\u003cbr\u003eConex\u00f5es: 6","BLZ\u003cbr\u003eConex\u00f5es: 5","MDG\u003cbr\u003eConex\u00f5es: 24","MWI\u003cbr\u003eConex\u00f5es: 4","ZWE\u003cbr\u003eConex\u00f5es: 18","MSR\u003cbr\u003eConex\u00f5es: 2","ATG\u003cbr\u003eConex\u00f5es: 9","MLI\u003cbr\u003eConex\u00f5es: 10","BWA\u003cbr\u003eConex\u00f5es: 10","VGB\u003cbr\u003eConex\u00f5es: 7","LSO\u003cbr\u003eConex\u00f5es: 5","MRT\u003cbr\u003eConex\u00f5es: 9","SWZ\u003cbr\u003eConex\u00f5es: 11","ZMB\u003cbr\u003eConex\u00f5es: 12","NPL\u003cbr\u003eConex\u00f5es: 6","TJK\u003cbr\u003eConex\u00f5es: 8","BTN\u003cbr\u003eConex\u00f5es: 2","TTO\u003cbr\u003eConex\u00f5es: 37","GRD\u003cbr\u003eConex\u00f5es: 10","VCT\u003cbr\u003eConex\u00f5es: 6","ETH\u003cbr\u003eConex\u00f5es: 17","MHL\u003cbr\u003eConex\u00f5es: 5","SOM\u003cbr\u003eConex\u00f5es: 5","SYR\u003cbr\u003eConex\u00f5es: 4","UZB\u003cbr\u003eConex\u00f5es: 8","TCD\u003cbr\u003eConex\u00f5es: 5","KIR\u003cbr\u003eConex\u00f5es: 1","WSM\u003cbr\u003eConex\u00f5es: 4","BHS\u003cbr\u003eConex\u00f5es: 7","BRB\u003cbr\u003eConex\u00f5es: 7","DJI\u003cbr\u003eConex\u00f5es: 7","SXM\u003cbr\u003eConex\u00f5es: 2","BES\u003cbr\u003eConex\u00f5es: 2","LCA\u003cbr\u003eConex\u00f5es: 7","RWA\u003cbr\u003eConex\u00f5es: 8","CAF\u003cbr\u003eConex\u00f5es: 5","GMB\u003cbr\u003eConex\u00f5es: 3","PSE\u003cbr\u003eConex\u00f5es: 5","LBR\u003cbr\u003eConex\u00f5es: 10","NER\u003cbr\u003eConex\u00f5es: 4","TON\u003cbr\u003eConex\u00f5es: 3","CYM\u003cbr\u003eConex\u00f5es: 8","SPM\u003cbr\u003eConex\u00f5es: 1","DMA\u003cbr\u003eConex\u00f5es: 3","KNA\u003cbr\u003eConex\u00f5es: 3","AIA\u003cbr\u003eConex\u00f5es: 2","GUM\u003cbr\u003eConex\u00f5es: 5","TLS\u003cbr\u003eConex\u00f5es: 2","PLW\u003cbr\u003eConex\u00f5es: 1","BDI\u003cbr\u003eConex\u00f5es: 2","BLM\u003cbr\u003eConex\u00f5es: 2","TCA\u003cbr\u003eConex\u00f5es: 4","SSD\u003cbr\u003eConex\u00f5es: 2","ASM\u003cbr\u003eConex\u00f5es: 1","CXR\u003cbr\u003eConex\u00f5es: 1","COM\u003cbr\u003eConex\u00f5es: 1","GNB\u003cbr\u003eConex\u00f5es: 2","NRU\u003cbr\u003eConex\u00f5es: 1","SLB\u003cbr\u003eConex\u00f5es: 2","COK\u003cbr\u003eConex\u00f5es: 1","TUV\u003cbr\u003eConex\u00f5es: 1","CPV\u003cbr\u003eConex\u00f5es: 4","SHN\u003cbr\u003eConex\u00f5es: 2","STP\u003cbr\u003eConex\u00f5es: 2","FSM\u003cbr\u003eConex\u00f5es: 1","ATA\u003cbr\u003eConex\u00f5es: 1","NFK\u003cbr\u003eConex\u00f5es: 1"],"visible":true,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null,22.396428,35.86166,null,22.396428,37.09024,null,35.86166,36.204824,null,23.634501,37.09024,null,22.396428,36.204824,null,14.058324,37.09024,null,22.396428,51.165691,null,35.86166,51.165691,null,35.86166,55.378051,null,52.132633,51.165691,null,37.09024,56.130366,null,35.86166,52.132633,null,35.86166,35.907757,null,35.86166,22.396428,null,22.396428,-0.789275,null,22.396428,20.593684,null,22.396428,55.378051,null,22.396428,50.850346,null,35.86166,56.130366,null,22.396428,56.130366,null,22.396428,46.603354,null,35.86166,-25.274398,null,35.86166,46.603354,null,51.165691,51.919438,null,22.396428,49.817492,null,22.396428,49.815273,null,18.735693,37.09024,null,22.396428,-30.559482,null,22.396428,23.634501,null,37.09024,50.850346,null,22.396428,52.132633,null,51.165691,46.603354,null,22.396428,40.463667,null,22.396428,14.058324,null,35.86166,14.058324,null,35.86166,40.463667,null,51.165691,52.132633,null,51.165691,47.516231,null,35.86166,41.87194,null,35.86166,4.210484,null,22.396428,41.87194,null,52.132633,50.850346,null,22.396428,23.424076,null,51.165691,41.87194,null,35.86166,50.850346,null,52.132633,46.603354,null,22.396428,26.3351,null,14.058324,51.165691,null,35.86166,1.352083,null,14.058324,36.204824,null,15.870032,51.165691,null,35.86166,51.919438,null,22.396428,22.198745,null,51.165691,46.818188,null,35.86166,63.397768,null,35.86166,20.593684,null,38.963745,37.09024,null,51.165691,49.817492,null,35.86166,23.634501,null,51.165691,40.463667,null,35.86166,33.223191,null,35.86166,23.885942,null,38.963745,51.165691,null,22.396428,4.210484,null,35.86166,23.424076,null,22.396428,-25.274398,null,51.165691,55.378051,null,35.86166,-35.675147,null,35.86166,61.52401,null,35.86166,-14.235004,null,52.132633,40.463667,null,22.396428,23.685,null,51.165691,63.397768,null,37.09024,-25.274398,null,15.870032,37.09024,null,50.850346,46.603354,null,38.963745,52.132633,null,22.396428,56.26392,null,35.86166,-0.789275,null,14.058324,35.907757,null,63.397768,60.472024,null,15.870032,49.817492,null,35.86166,12.879721,null,35.86166,15.870032,null,52.132633,41.87194,null,38.963745,47.411631,null,35.86166,56.26392,null,22.396428,35.907757,null,35.86166,-30.559482,null,40.463667,39.399872,null,14.058324,41.87194,null,52.132633,55.378051,null,37.09024,-38.416097,null,35.86166,32.427908,null,63.397768,56.26392,null,37.09024,55.378051,null,63.397768,61.92411,null,14.058324,55.378051,null,22.396428,12.565679,null,23.885942,29.31166,null,22.396428,7.539989,null,51.165691,37.09024,null,-0.789275,36.204824,null,51.165691,50.850346,null,37.09024,36.204824,null,35.86166,60.472024,null,22.396428,63.397768,null,22.396428,1.352083,null,51.165691,56.26392,null,35.86166,31.046051,null,22.396428,46.818188,null,38.963745,33.223191,null,14.058324,40.463667,null,14.058324,52.132633,null,15.870032,-25.274398,null,22.396428,58.595272,null,40.463667,46.603354,null,22.396428,30.375321,null,40.463667,41.87194,null,14.058324,63.397768,null,35.86166,-40.900557,null,39.399872,55.378051,null,14.058324,46.603354,null,52.132633,51.919438,null,14.058324,51.919438,null,37.09024,-30.559482,null,1.352083,35.86166,null,51.165691,53.41291,null,15.870032,36.204824,null,22.396428,39.399872,null,35.86166,39.074208,null,40.463667,51.165691,null,22.396428,23.885942,null,51.165691,23.424076,null,35.86166,38.963745,null,35.907757,35.86166,null,22.396428,33.886917,null,50.850346,52.132633,null,1.373333,55.378051,null,35.86166,9.081999,null,22.396428,38.963745,null,35.907757,14.058324,null,51.165691,47.162494,null,35.86166,39.399872,null,22.396428,51.919438,null,14.058324,35.86166,null,35.86166,4.570868,null,35.86166,46.818188,null,37.09024,25.03428,null,35.86166,47.516231,null,51.165691,61.92411,null,35.86166,21.916221,null,51.165691,35.86166,null,35.907757,36.204824,null,37.09024,1.352083,null,35.86166,48.379433,null,35.86166,61.92411,null,37.09024,52.132633,null,38.963745,55.378051,null,38.963745,46.603354,null,15.870032,1.352083,null,47.411631,45.943161,null,23.634501,15.783471,null,55.169438,63.397768,null,37.09024,21.916221,null,51.165691,45.943161,null,35.86166,46.151241,null,35.86166,23.685,null,35.86166,-9.189967,null,38.963745,51.919438,null,38.963745,39.074208,null,42.733883,52.132633,null,35.86166,26.820553,null,35.86166,48.019573,null,37.09024,35.86166,null,14.058324,47.516231,null,51.165691,26.820553,null,22.396428,60.472024,null,35.86166,49.817492,null,37.09024,8.537981,null,15.870032,-0.789275,null,39.399872,40.463667,null,38.963745,63.397768,null,52.132633,49.817492,null,37.09024,-35.675147,null,63.397768,51.919438,null,38.963745,40.463667,null,22.396428,15.870032,null,35.86166,28.0,null,38.963745,48.669026,null,51.165691,38.963745,null,14.058324,56.879635,null,37.09024,22.396428,null,50.850346,51.165691,null,40.463667,31.791702,null,22.396428,47.516231,null,22.396428,15.783471,null,37.09024,20.593684,null,61.52401,53.709807,null,22.396428,61.92411,null,29.31166,30.585164,null,29.31166,21.512583,null,37.09024,-9.189967,null,22.396428,7.873054,null,63.397768,50.850346,null,56.879635,63.397768,null,15.870032,52.132633,null,51.165691,60.472024,null,29.31166,25.354826,null,35.86166,12.862807,null,35.86166,53.41291,null,50.850346,55.378051,null,52.132633,37.09024,null,14.058324,23.424076,null,51.165691,61.52401,null,35.86166,8.537981,null,15.870032,50.850346,null,51.165691,48.669026,null,51.165691,55.169438,null,35.86166,45.943161,null,14.058324,56.130366,null,35.86166,33.854721,null,37.09024,31.046051,null,35.86166,30.375321,null,53.41291,36.204824,null,22.396428,61.52401,null,52.132633,63.397768,null,14.058324,-25.274398,null,35.86166,30.585164,null,38.963745,56.879635,null,35.86166,-38.416097,null,1.352083,37.09024,null,51.165691,46.151241,null,35.86166,11.825138,null,48.669026,46.603354,null,1.352083,-0.789275,null,35.86166,29.31166,null,52.132633,53.41291,null,22.396428,12.879721,null,39.399872,-30.559482,null,50.850346,49.815273,null,-30.559482,-19.015438,null,63.397768,51.165691,null,14.058324,33.223191,null,63.397768,37.09024,null,38.963745,31.791702,null,35.86166,56.879635,null,37.09024,41.87194,null,52.132633,23.424076,null,35.907757,1.352083,null,61.52401,48.019573,null,15.870032,63.397768,null,51.165691,49.815273,null,52.132633,56.130366,null,42.733883,51.165691,null,14.058324,61.52401,null,53.41291,35.86166,null,35.86166,25.354826,null,63.397768,55.378051,null,35.907757,37.09024,null,22.396428,48.669026,null,47.411631,61.52401,null,35.86166,-0.023559,null,35.86166,12.565679,null,51.165691,39.399872,null,35.86166,18.735693,null,38.963745,23.634501,null,-30.559482,55.378051,null,35.907757,-0.789275,null,23.885942,25.354826,null,48.669026,51.165691,null,37.09024,23.885942,null,51.165691,28.0,null,1.352083,35.907757,null,46.818188,51.165691,null,22.396428,18.735693,null,52.132633,47.516231,null,37.09024,46.818188,null,37.09024,12.865416,null,35.86166,-6.369028,null,-0.789275,-25.274398,null,22.396428,-17.713371,null,40.463667,55.378051,null,35.86166,9.748917,null,37.09024,23.424076,null,37.09024,35.907757,null,35.86166,31.791702,null,55.169438,51.919438,null,22.396428,53.41291,null,38.963745,42.733883,null,38.963745,23.885942,null,35.907757,25.354826,null,1.352083,22.396428,null,37.09024,9.748917,null,35.86166,7.946527,null,52.132633,61.92411,null,21.512583,33.223191,null,35.907757,12.565679,null,35.86166,55.169438,null,-30.559482,-18.665695,null,22.396428,36.140751,null,29.31166,23.424076,null,37.09024,4.210484,null,1.352083,-25.274398,null,51.165691,-30.559482,null,35.86166,47.162494,null,52.132633,56.26392,null,52.132633,-25.274398,null,37.09024,13.794185,null,48.669026,55.378051,null,1.352083,4.210484,null,41.608635,52.132633,null,22.396428,-29.609988,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.5020828278646496},"lon":[104.195397,-95.712891,null,114.109497,104.195397,null,114.109497,-95.712891,null,104.195397,138.252924,null,-102.552784,-95.712891,null,114.109497,138.252924,null,108.277199,-95.712891,null,114.109497,10.451526,null,104.195397,10.451526,null,104.195397,-3.435973,null,5.291266,10.451526,null,-95.712891,-106.346771,null,104.195397,5.291266,null,104.195397,127.766922,null,104.195397,114.109497,null,114.109497,113.921327,null,114.109497,78.96288,null,114.109497,-3.435973,null,114.109497,4.351721,null,104.195397,-106.346771,null,114.109497,-106.346771,null,114.109497,1.888334,null,104.195397,133.775136,null,104.195397,1.888334,null,10.451526,19.145136,null,114.109497,15.472962,null,114.109497,6.129583,null,-70.162651,-95.712891,null,114.109497,22.937506,null,114.109497,-102.552784,null,-95.712891,4.351721,null,114.109497,5.291266,null,10.451526,1.888334,null,114.109497,-3.74922,null,114.109497,108.277199,null,104.195397,108.277199,null,104.195397,-3.74922,null,10.451526,5.291266,null,10.451526,14.550072,null,104.195397,12.56738,null,104.195397,101.975766,null,114.109497,12.56738,null,5.291266,4.351721,null,114.109497,53.847818,null,10.451526,12.56738,null,104.195397,4.351721,null,5.291266,1.888334,null,114.109497,17.228331,null,108.277199,10.451526,null,104.195397,103.819836,null,108.277199,138.252924,null,100.992541,10.451526,null,104.195397,19.145136,null,114.109497,113.543873,null,10.451526,8.227512,null,104.195397,16.354896,null,104.195397,78.96288,null,35.243322,-95.712891,null,10.451526,15.472962,null,104.195397,-102.552784,null,10.451526,-3.74922,null,104.195397,43.679291,null,104.195397,45.079162,null,35.243322,10.451526,null,114.109497,101.975766,null,104.195397,53.847818,null,114.109497,133.775136,null,10.451526,-3.435973,null,104.195397,-71.542969,null,104.195397,105.318756,null,104.195397,-51.92528,null,5.291266,-3.74922,null,114.109497,90.3563,null,10.451526,16.354896,null,-95.712891,133.775136,null,100.992541,-95.712891,null,4.351721,1.888334,null,35.243322,5.291266,null,114.109497,9.501785,null,104.195397,113.921327,null,108.277199,127.766922,null,16.354896,8.468946,null,100.992541,15.472962,null,104.195397,121.774017,null,104.195397,100.992541,null,5.291266,12.56738,null,35.243322,28.369885,null,104.195397,9.501785,null,114.109497,127.766922,null,104.195397,22.937506,null,-3.74922,-8.224454,null,108.277199,12.56738,null,5.291266,-3.435973,null,-95.712891,-63.616672,null,104.195397,53.688046,null,16.354896,9.501785,null,-95.712891,-3.435973,null,16.354896,25.748151,null,108.277199,-3.435973,null,114.109497,104.990963,null,45.079162,47.481766,null,114.109497,-5.54708,null,10.451526,-95.712891,null,113.921327,138.252924,null,10.451526,4.351721,null,-95.712891,138.252924,null,104.195397,8.468946,null,114.109497,16.354896,null,114.109497,103.819836,null,10.451526,9.501785,null,104.195397,34.851612,null,114.109497,8.227512,null,35.243322,43.679291,null,108.277199,-3.74922,null,108.277199,5.291266,null,100.992541,133.775136,null,114.109497,25.013607,null,-3.74922,1.888334,null,114.109497,69.345116,null,-3.74922,12.56738,null,108.277199,16.354896,null,104.195397,174.885971,null,-8.224454,-3.435973,null,108.277199,1.888334,null,5.291266,19.145136,null,108.277199,19.145136,null,-95.712891,22.937506,null,103.819836,104.195397,null,10.451526,-8.24389,null,100.992541,138.252924,null,114.109497,-8.224454,null,104.195397,21.824312,null,-3.74922,10.451526,null,114.109497,45.079162,null,10.451526,53.847818,null,104.195397,35.243322,null,127.766922,104.195397,null,114.109497,9.537499,null,4.351721,5.291266,null,32.290275,-3.435973,null,104.195397,8.675277,null,114.109497,35.243322,null,127.766922,108.277199,null,10.451526,19.503304,null,104.195397,-8.224454,null,114.109497,19.145136,null,108.277199,104.195397,null,104.195397,-74.297333,null,104.195397,8.227512,null,-95.712891,-77.39628,null,104.195397,14.550072,null,10.451526,25.748151,null,104.195397,95.955974,null,10.451526,104.195397,null,127.766922,138.252924,null,-95.712891,103.819836,null,104.195397,31.16558,null,104.195397,25.748151,null,-95.712891,5.291266,null,35.243322,-3.435973,null,35.243322,1.888334,null,100.992541,103.819836,null,28.369885,24.96676,null,-102.552784,-90.230759,null,23.881275,16.354896,null,-95.712891,95.955974,null,10.451526,24.96676,null,104.195397,14.995463,null,104.195397,90.3563,null,104.195397,-75.015152,null,35.243322,19.145136,null,35.243322,21.824312,null,25.48583,5.291266,null,104.195397,30.802498,null,104.195397,66.923684,null,-95.712891,104.195397,null,108.277199,14.550072,null,10.451526,30.802498,null,114.109497,8.468946,null,104.195397,15.472962,null,-95.712891,-80.782127,null,100.992541,113.921327,null,-8.224454,-3.74922,null,35.243322,16.354896,null,5.291266,15.472962,null,-95.712891,-71.542969,null,16.354896,19.145136,null,35.243322,-3.74922,null,114.109497,100.992541,null,104.195397,3.0,null,35.243322,19.699024,null,10.451526,35.243322,null,108.277199,24.603189,null,-95.712891,114.109497,null,4.351721,10.451526,null,-3.74922,-7.09262,null,114.109497,14.550072,null,114.109497,-90.230759,null,-95.712891,78.96288,null,105.318756,27.953389,null,114.109497,25.748151,null,47.481766,36.238414,null,47.481766,55.923255,null,-95.712891,-75.015152,null,114.109497,80.771797,null,16.354896,4.351721,null,24.603189,16.354896,null,100.992541,5.291266,null,10.451526,8.468946,null,47.481766,51.183884,null,104.195397,30.217636,null,104.195397,-8.24389,null,4.351721,-3.435973,null,5.291266,-95.712891,null,108.277199,53.847818,null,10.451526,105.318756,null,104.195397,-80.782127,null,100.992541,4.351721,null,10.451526,19.699024,null,10.451526,23.881275,null,104.195397,24.96676,null,108.277199,-106.346771,null,104.195397,35.862285,null,-95.712891,34.851612,null,104.195397,69.345116,null,-8.24389,138.252924,null,114.109497,105.318756,null,5.291266,16.354896,null,108.277199,133.775136,null,104.195397,36.238414,null,35.243322,24.603189,null,104.195397,-63.616672,null,103.819836,-95.712891,null,10.451526,14.995463,null,104.195397,42.590275,null,19.699024,1.888334,null,103.819836,113.921327,null,104.195397,47.481766,null,5.291266,-8.24389,null,114.109497,121.774017,null,-8.224454,22.937506,null,4.351721,6.129583,null,22.937506,29.154857,null,16.354896,10.451526,null,108.277199,43.679291,null,16.354896,-95.712891,null,35.243322,-7.09262,null,104.195397,24.603189,null,-95.712891,12.56738,null,5.291266,53.847818,null,127.766922,103.819836,null,105.318756,66.923684,null,100.992541,16.354896,null,10.451526,6.129583,null,5.291266,-106.346771,null,25.48583,10.451526,null,108.277199,105.318756,null,-8.24389,104.195397,null,104.195397,51.183884,null,16.354896,-3.435973,null,127.766922,-95.712891,null,114.109497,19.699024,null,28.369885,105.318756,null,104.195397,37.906193,null,104.195397,104.990963,null,10.451526,-8.224454,null,104.195397,-70.162651,null,35.243322,-102.552784,null,22.937506,-3.435973,null,127.766922,113.921327,null,45.079162,51.183884,null,19.699024,10.451526,null,-95.712891,45.079162,null,10.451526,3.0,null,103.819836,127.766922,null,8.227512,10.451526,null,114.109497,-70.162651,null,5.291266,14.550072,null,-95.712891,8.227512,null,-95.712891,-85.207229,null,104.195397,34.888822,null,113.921327,133.775136,null,114.109497,178.065033,null,-3.74922,-3.435973,null,104.195397,-83.753428,null,-95.712891,53.847818,null,-95.712891,127.766922,null,104.195397,-7.09262,null,23.881275,19.145136,null,114.109497,-8.24389,null,35.243322,25.48583,null,35.243322,45.079162,null,127.766922,51.183884,null,103.819836,114.109497,null,-95.712891,-83.753428,null,104.195397,-1.023194,null,5.291266,25.748151,null,55.923255,43.679291,null,127.766922,104.990963,null,104.195397,23.881275,null,22.937506,35.529562,null,114.109497,-5.353585,null,47.481766,53.847818,null,-95.712891,101.975766,null,103.819836,133.775136,null,10.451526,22.937506,null,104.195397,19.503304,null,5.291266,9.501785,null,5.291266,133.775136,null,-95.712891,-88.89653,null,19.699024,-3.435973,null,103.819836,101.975766,null,21.745275,5.291266,null,114.109497,28.233608,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[-11.202692,50.850346,33.854721,39.399872,14.497401,26.02751,25.354826,23.885942,23.424076,55.378051,33.0,41.0,42.546245,-25.274398,47.516231,43.915886,-14.235004,42.733883,56.130366,-35.675147,35.86166,4.570868,-11.6455,-0.228021,-2.1646,45.1,35.126413,49.817492,9.30769,56.26392,1.650801,58.595272,61.92411,46.603354,-17.679742,11.825138,-0.803689,51.165691,36.140751,39.074208,22.396428,47.162494,64.963051,53.41291,31.046051,41.87194,36.204824,30.585164,-0.023559,35.907757,29.31166,56.879635,26.3351,55.169438,49.815273,4.210484,35.937496,21.00789,31.791702,52.132633,17.607789,60.472024,30.375321,8.537981,12.879721,51.919438,45.943161,61.52401,43.94236,44.016521,20.593684,1.352083,48.669026,46.151241,-30.559482,40.463667,63.397768,46.818188,38.861034,15.870032,8.619543,33.886917,38.963745,48.379433,41.608635,-6.369028,37.09024,41.377491,-16.290154,-38.416097,23.634501,9.748917,13.794185,4.860416,-23.442503,-9.189967,-32.522779,53.709807,42.315407,17.570692,12.565679,18.109581,12.865416,46.946947,-4.679574,34.802075,-1.831239,-18.766947,28.0,23.685,40.069099,13.193887,-22.328474,4.535277,21.916221,7.369722,16.5388,6.611111,7.873054,15.454166,21.521757,15.414999,18.735693,9.145,15.179384,-17.713371,13.443182,31.952162,7.946527,15.783471,9.945587,18.971187,15.199999,-0.789275,32.427908,33.223191,7.539989,48.019573,40.339852,41.20438,19.85627,6.428055,22.198745,-13.254308,3.202778,-20.348404,46.862496,42.708678,-18.665695,21.512583,-22.95764,28.394857,12.16957,12.52111,-20.904305,-15.376706,-40.900557,9.081999,-6.314993,-8.874217,-1.940278,13.909444,8.460555,14.058324,5.152149,-19.015438,12.862807,3.919305,-21.178986,10.691803,1.373333,26.820553,12.238333,6.42375,-13.759029,15.552727,-13.133897,40.143105,17.060816,25.03428,61.892635,47.411631,38.969719,13.444304,17.189877,11.803749,17.897476,0.18636,-29.609988,-15.965,-26.522503,12.262776,17.357822,18.420695,19.3133,18.04248,21.694025,71.706936,27.514162,7.425554,7.862684,-9.64571,-0.522778,32.3078,12.20189,16.742498,-90.0,-19.054445,-21.236736,18.218785,7.131474,12.984305,-3.370417,7.51498,-12.164165,-3.373056,17.664332,-7.109535,-14.28522,-90.0],"lon":[17.873887,4.351721,35.862285,-8.224454,-14.452362,50.55096,51.183884,45.079162,53.847818,-3.435973,65.0,20.0,1.601554,133.775136,14.550072,17.679076,-51.92528,25.48583,-106.346771,-71.542969,104.195397,-74.297333,43.3333,15.827659,24.15536,15.2,33.429859,15.472962,2.315834,9.501785,10.267895,25.013607,25.748151,1.888334,-149.406843,42.590275,11.609444,10.451526,-5.353585,21.824312,114.109497,19.503304,-19.020835,-8.24389,34.851612,12.56738,138.252924,36.238414,37.906193,127.766922,47.481766,24.603189,17.228331,23.881275,6.129583,101.975766,14.375416,-10.940835,-7.09262,5.291266,8.081666,8.468946,69.345116,-80.782127,121.774017,19.145136,24.96676,105.318756,12.457777,21.005859,78.96288,103.819836,19.699024,14.995463,22.937506,-3.74922,16.354896,8.227512,71.276093,100.992541,0.824782,9.537499,35.243322,31.16558,21.745275,34.888822,-95.712891,64.585262,-63.588653,-63.616672,-102.552784,-83.753428,-88.89653,-58.93018,-58.443832,-75.015152,-55.765835,27.953389,43.356892,-3.996166,104.990963,-77.297508,-85.207229,-56.32509,55.491977,38.996815,-78.183406,46.869107,3.0,90.3563,45.038189,-59.543198,24.684866,114.727669,95.955974,12.354722,-23.0418,20.939444,80.771797,18.732207,-77.781167,-61.370976,-70.162651,40.489673,39.782334,178.065033,-15.310139,35.233154,-1.023194,-90.230759,-9.696645,-72.285215,-86.241905,113.921327,53.688046,43.679291,-5.54708,66.923684,127.510093,74.766098,102.495496,-9.429499,113.543873,34.301525,73.22068,57.552152,103.846656,19.37439,35.529562,55.923255,18.49041,84.124008,-68.990021,-69.968338,165.618042,166.959158,174.885971,8.675277,143.95555,125.727,29.873888,-60.978893,-11.779889,108.277199,46.199616,29.154857,30.217636,-56.027783,-175.198242,-61.222503,32.290275,30.802498,-1.561593,-66.58973,-172.104629,48.516388,27.849332,47.576927,-61.796428,-77.39628,-6.911806,28.369885,59.556278,144.793731,-88.49765,-15.180413,-62.83055,6.613081,28.233608,-5.7089,31.465866,-61.604171,-62.782998,-64.639968,-81.2546,-63.05483,-71.797928,-42.604303,90.433601,150.550812,30.217636,160.156194,166.931503,-64.7505,-68.262383,-62.187366,0.0,-169.867233,-159.777671,-63.043653,171.184478,-61.287228,-168.734039,134.58252,96.870956,29.918886,145.94351,179.194167,-170.70444,0.0],"marker":{"color":"red","opacity":0.8,"size":[15.6,20.0,8.4,20.0,4.0,12.4,8.4,12.0,12.8,15.6,4.0,6.4,4.8,12.0,11.2,5.6,20.0,20.0,20.0,17.6,20.0,9.6,1.2,6.8,6.8,7.6,10.4,10.4,4.0,12.0,4.4,8.4,8.8,15.2,4.8,2.0,5.6,20.0,2.4,11.2,20.0,9.2,6.0,20.0,9.6,12.8,12.8,8.0,8.0,20.0,11.6,20.0,4.0,20.0,6.0,9.6,8.8,4.0,8.4,20.0,1.6,10.0,6.4,10.4,12.0,10.4,10.8,20.0,2.0,8.4,12.0,20.0,20.0,7.2,20.0,20.0,20.0,20.0,3.2,20.0,4.4,6.0,20.0,8.8,12.8,6.0,20.0,4.4,11.2,7.6,20.0,9.6,12.4,3.6,6.4,20.0,8.4,6.4,16.0,5.2,5.6,4.8,7.6,1.2,6.4,2.8,7.2,12.0,6.0,5.2,6.4,4.0,2.0,4.0,4.8,6.0,2.4,1.6,5.6,0.8,6.0,2.0,15.6,6.8,1.2,10.4,2.0,2.0,10.4,12.8,4.4,3.2,6.0,20.0,8.0,5.6,12.0,15.6,1.6,4.8,5.6,3.6,6.0,2.8,4.8,16.0,4.8,5.2,4.4,10.4,4.0,3.6,5.2,4.8,4.4,2.4,10.4,7.2,3.6,2.0,4.8,2.8,4.0,20.0,2.0,4.8,4.0,4.0,0.8,16.0,7.2,9.2,4.4,6.0,1.2,2.8,4.8,7.2,2.8,3.2,2.0,12.4,5.2,1.6,4.0,1.2,0.4,2.8,3.2,0.8,4.0,3.6,3.2,2.4,2.4,2.4,2.4,3.2,2.0,0.8,2.0,2.0,0.4,2.4,2.0,0.8,0.4,0.4,2.4,1.2,2.4,2.0,0.8,1.2,0.4,1.2,1.2,0.8,0.4,0.4]},"mode":"markers","name":"2016","text":["AGO\u003cbr\u003eConex\u00f5es: 39","BEL\u003cbr\u003eConex\u00f5es: 186","LBN\u003cbr\u003eConex\u00f5es: 21","PRT\u003cbr\u003eConex\u00f5es: 147","SEN\u003cbr\u003eConex\u00f5es: 10","BHR\u003cbr\u003eConex\u00f5es: 31","QAT\u003cbr\u003eConex\u00f5es: 21","SAU\u003cbr\u003eConex\u00f5es: 30","ARE\u003cbr\u003eConex\u00f5es: 32","GBR\u003cbr\u003eConex\u00f5es: 39","AFG\u003cbr\u003eConex\u00f5es: 10","ALB\u003cbr\u003eConex\u00f5es: 16","AND\u003cbr\u003eConex\u00f5es: 12","AUS\u003cbr\u003eConex\u00f5es: 30","AUT\u003cbr\u003eConex\u00f5es: 28","BIH\u003cbr\u003eConex\u00f5es: 14","BRA\u003cbr\u003eConex\u00f5es: 98","BGR\u003cbr\u003eConex\u00f5es: 66","CAN\u003cbr\u003eConex\u00f5es: 100","CHL\u003cbr\u003eConex\u00f5es: 44","CHN\u003cbr\u003eConex\u00f5es: 230","COL\u003cbr\u003eConex\u00f5es: 24","COM\u003cbr\u003eConex\u00f5es: 3","COG\u003cbr\u003eConex\u00f5es: 17","COD\u003cbr\u003eConex\u00f5es: 17","HRV\u003cbr\u003eConex\u00f5es: 19","CYP\u003cbr\u003eConex\u00f5es: 26","CZE\u003cbr\u003eConex\u00f5es: 26","BEN\u003cbr\u003eConex\u00f5es: 10","DNK\u003cbr\u003eConex\u00f5es: 30","GNQ\u003cbr\u003eConex\u00f5es: 11","EST\u003cbr\u003eConex\u00f5es: 21","FIN\u003cbr\u003eConex\u00f5es: 22","FRA\u003cbr\u003eConex\u00f5es: 38","PYF\u003cbr\u003eConex\u00f5es: 12","DJI\u003cbr\u003eConex\u00f5es: 5","GAB\u003cbr\u003eConex\u00f5es: 14","DEU\u003cbr\u003eConex\u00f5es: 213","GIB\u003cbr\u003eConex\u00f5es: 6","GRC\u003cbr\u003eConex\u00f5es: 28","HKG\u003cbr\u003eConex\u00f5es: 153","HUN\u003cbr\u003eConex\u00f5es: 23","ISL\u003cbr\u003eConex\u00f5es: 15","IRL\u003cbr\u003eConex\u00f5es: 68","ISR\u003cbr\u003eConex\u00f5es: 24","ITA\u003cbr\u003eConex\u00f5es: 32","JPN\u003cbr\u003eConex\u00f5es: 32","JOR\u003cbr\u003eConex\u00f5es: 20","KEN\u003cbr\u003eConex\u00f5es: 20","KOR\u003cbr\u003eConex\u00f5es: 105","KWT\u003cbr\u003eConex\u00f5es: 29","LVA\u003cbr\u003eConex\u00f5es: 72","LBY\u003cbr\u003eConex\u00f5es: 10","LTU\u003cbr\u003eConex\u00f5es: 83","LUX\u003cbr\u003eConex\u00f5es: 15","MYS\u003cbr\u003eConex\u00f5es: 24","MLT\u003cbr\u003eConex\u00f5es: 22","MRT\u003cbr\u003eConex\u00f5es: 10","MAR\u003cbr\u003eConex\u00f5es: 21","NLD\u003cbr\u003eConex\u00f5es: 184","NER\u003cbr\u003eConex\u00f5es: 4","NOR\u003cbr\u003eConex\u00f5es: 25","PAK\u003cbr\u003eConex\u00f5es: 16","PAN\u003cbr\u003eConex\u00f5es: 26","PHL\u003cbr\u003eConex\u00f5es: 30","POL\u003cbr\u003eConex\u00f5es: 26","ROU\u003cbr\u003eConex\u00f5es: 27","RUS\u003cbr\u003eConex\u00f5es: 117","SMR\u003cbr\u003eConex\u00f5es: 5","SRB\u003cbr\u003eConex\u00f5es: 21","IND\u003cbr\u003eConex\u00f5es: 30","SGP\u003cbr\u003eConex\u00f5es: 110","SVK\u003cbr\u003eConex\u00f5es: 92","SVN\u003cbr\u003eConex\u00f5es: 18","ZAF\u003cbr\u003eConex\u00f5es: 146","ESP\u003cbr\u003eConex\u00f5es: 167","SWE\u003cbr\u003eConex\u00f5es: 147","CHE\u003cbr\u003eConex\u00f5es: 170","TJK\u003cbr\u003eConex\u00f5es: 8","THA\u003cbr\u003eConex\u00f5es: 135","TGO\u003cbr\u003eConex\u00f5es: 11","TUN\u003cbr\u003eConex\u00f5es: 15","TUR\u003cbr\u003eConex\u00f5es: 163","UKR\u003cbr\u003eConex\u00f5es: 22","MKD\u003cbr\u003eConex\u00f5es: 32","TZA\u003cbr\u003eConex\u00f5es: 15","USA\u003cbr\u003eConex\u00f5es: 191","UZB\u003cbr\u003eConex\u00f5es: 11","BOL\u003cbr\u003eConex\u00f5es: 28","ARG\u003cbr\u003eConex\u00f5es: 19","MEX\u003cbr\u003eConex\u00f5es: 55","CRI\u003cbr\u003eConex\u00f5es: 24","SLV\u003cbr\u003eConex\u00f5es: 31","GUY\u003cbr\u003eConex\u00f5es: 9","PRY\u003cbr\u003eConex\u00f5es: 16","PER\u003cbr\u003eConex\u00f5es: 72","URY\u003cbr\u003eConex\u00f5es: 21","BLR\u003cbr\u003eConex\u00f5es: 16","GEO\u003cbr\u003eConex\u00f5es: 40","MLI\u003cbr\u003eConex\u00f5es: 13","KHM\u003cbr\u003eConex\u00f5es: 14","JAM\u003cbr\u003eConex\u00f5es: 12","NIC\u003cbr\u003eConex\u00f5es: 19","SPM\u003cbr\u003eConex\u00f5es: 3","SYC\u003cbr\u003eConex\u00f5es: 16","SYR\u003cbr\u003eConex\u00f5es: 7","ECU\u003cbr\u003eConex\u00f5es: 18","MDG\u003cbr\u003eConex\u00f5es: 30","DZA\u003cbr\u003eConex\u00f5es: 15","BGD\u003cbr\u003eConex\u00f5es: 13","ARM\u003cbr\u003eConex\u00f5es: 16","BRB\u003cbr\u003eConex\u00f5es: 10","BWA\u003cbr\u003eConex\u00f5es: 5","BRN\u003cbr\u003eConex\u00f5es: 10","MMR\u003cbr\u003eConex\u00f5es: 12","CMR\u003cbr\u003eConex\u00f5es: 15","CPV\u003cbr\u003eConex\u00f5es: 6","CAF\u003cbr\u003eConex\u00f5es: 4","LKA\u003cbr\u003eConex\u00f5es: 14","TCD\u003cbr\u003eConex\u00f5es: 2","CUB\u003cbr\u003eConex\u00f5es: 15","DMA\u003cbr\u003eConex\u00f5es: 5","DOM\u003cbr\u003eConex\u00f5es: 39","ETH\u003cbr\u003eConex\u00f5es: 17","ERI\u003cbr\u003eConex\u00f5es: 3","FJI\u003cbr\u003eConex\u00f5es: 26","GMB\u003cbr\u003eConex\u00f5es: 5","PSE\u003cbr\u003eConex\u00f5es: 5","GHA\u003cbr\u003eConex\u00f5es: 26","GTM\u003cbr\u003eConex\u00f5es: 32","GIN\u003cbr\u003eConex\u00f5es: 11","HTI\u003cbr\u003eConex\u00f5es: 8","HND\u003cbr\u003eConex\u00f5es: 15","IDN\u003cbr\u003eConex\u00f5es: 74","IRN\u003cbr\u003eConex\u00f5es: 20","IRQ\u003cbr\u003eConex\u00f5es: 14","CIV\u003cbr\u003eConex\u00f5es: 30","KAZ\u003cbr\u003eConex\u00f5es: 39","PRK\u003cbr\u003eConex\u00f5es: 4","KGZ\u003cbr\u003eConex\u00f5es: 12","LAO\u003cbr\u003eConex\u00f5es: 14","LBR\u003cbr\u003eConex\u00f5es: 9","MAC\u003cbr\u003eConex\u00f5es: 15","MWI\u003cbr\u003eConex\u00f5es: 7","MDV\u003cbr\u003eConex\u00f5es: 12","MUS\u003cbr\u003eConex\u00f5es: 40","MNG\u003cbr\u003eConex\u00f5es: 12","MNE\u003cbr\u003eConex\u00f5es: 13","MOZ\u003cbr\u003eConex\u00f5es: 11","OMN\u003cbr\u003eConex\u00f5es: 26","NAM\u003cbr\u003eConex\u00f5es: 10","NPL\u003cbr\u003eConex\u00f5es: 9","CUW\u003cbr\u003eConex\u00f5es: 13","ABW\u003cbr\u003eConex\u00f5es: 12","NCL\u003cbr\u003eConex\u00f5es: 11","VUT\u003cbr\u003eConex\u00f5es: 6","NZL\u003cbr\u003eConex\u00f5es: 26","NGA\u003cbr\u003eConex\u00f5es: 18","PNG\u003cbr\u003eConex\u00f5es: 9","TLS\u003cbr\u003eConex\u00f5es: 5","RWA\u003cbr\u003eConex\u00f5es: 12","LCA\u003cbr\u003eConex\u00f5es: 7","SLE\u003cbr\u003eConex\u00f5es: 10","VNM\u003cbr\u003eConex\u00f5es: 89","SOM\u003cbr\u003eConex\u00f5es: 5","ZWE\u003cbr\u003eConex\u00f5es: 12","SDN\u003cbr\u003eConex\u00f5es: 10","SUR\u003cbr\u003eConex\u00f5es: 10","TON\u003cbr\u003eConex\u00f5es: 2","TTO\u003cbr\u003eConex\u00f5es: 40","UGA\u003cbr\u003eConex\u00f5es: 18","EGY\u003cbr\u003eConex\u00f5es: 23","BFA\u003cbr\u003eConex\u00f5es: 11","VEN\u003cbr\u003eConex\u00f5es: 15","WSM\u003cbr\u003eConex\u00f5es: 3","YEM\u003cbr\u003eConex\u00f5es: 7","ZMB\u003cbr\u003eConex\u00f5es: 12","AZE\u003cbr\u003eConex\u00f5es: 18","ATG\u003cbr\u003eConex\u00f5es: 7","BHS\u003cbr\u003eConex\u00f5es: 8","FRO\u003cbr\u003eConex\u00f5es: 5","MDA\u003cbr\u003eConex\u00f5es: 31","TKM\u003cbr\u003eConex\u00f5es: 13","GUM\u003cbr\u003eConex\u00f5es: 4","BLZ\u003cbr\u003eConex\u00f5es: 10","GNB\u003cbr\u003eConex\u00f5es: 3","BLM\u003cbr\u003eConex\u00f5es: 1","STP\u003cbr\u003eConex\u00f5es: 7","LSO\u003cbr\u003eConex\u00f5es: 8","SHN\u003cbr\u003eConex\u00f5es: 2","SWZ\u003cbr\u003eConex\u00f5es: 10","GRD\u003cbr\u003eConex\u00f5es: 9","KNA\u003cbr\u003eConex\u00f5es: 8","VGB\u003cbr\u003eConex\u00f5es: 6","CYM\u003cbr\u003eConex\u00f5es: 6","SXM\u003cbr\u003eConex\u00f5es: 6","TCA\u003cbr\u003eConex\u00f5es: 6","GRL\u003cbr\u003eConex\u00f5es: 8","BTN\u003cbr\u003eConex\u00f5es: 5","FSM\u003cbr\u003eConex\u00f5es: 2","SSD\u003cbr\u003eConex\u00f5es: 5","SLB\u003cbr\u003eConex\u00f5es: 5","NRU\u003cbr\u003eConex\u00f5es: 1","BMU\u003cbr\u003eConex\u00f5es: 6","BES\u003cbr\u003eConex\u00f5es: 5","MSR\u003cbr\u003eConex\u00f5es: 2","ATA\u003cbr\u003eConex\u00f5es: 1","NIU\u003cbr\u003eConex\u00f5es: 1","COK\u003cbr\u003eConex\u00f5es: 6","AIA\u003cbr\u003eConex\u00f5es: 3","MHL\u003cbr\u003eConex\u00f5es: 6","VCT\u003cbr\u003eConex\u00f5es: 5","KIR\u003cbr\u003eConex\u00f5es: 2","PLW\u003cbr\u003eConex\u00f5es: 3","CCK\u003cbr\u003eConex\u00f5es: 1","BDI\u003cbr\u003eConex\u00f5es: 3","MNP\u003cbr\u003eConex\u00f5es: 3","TUV\u003cbr\u003eConex\u00f5es: 2","ASM\u003cbr\u003eConex\u00f5es: 1","ATB\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null,22.396428,37.09024,null,22.396428,35.86166,null,22.396428,36.204824,null,35.86166,36.204824,null,23.634501,37.09024,null,12.879721,36.204824,null,20.593684,37.09024,null,22.396428,51.165691,null,35.86166,51.165691,null,37.09024,56.130366,null,14.058324,37.09024,null,51.165691,51.919438,null,35.86166,55.378051,null,52.132633,51.165691,null,22.396428,14.058324,null,22.396428,46.603354,null,35.86166,52.132633,null,35.86166,35.907757,null,35.86166,22.396428,null,35.86166,56.130366,null,35.86166,-25.274398,null,51.165691,46.603354,null,22.396428,55.378051,null,35.86166,46.603354,null,22.396428,1.352083,null,35.86166,4.210484,null,18.735693,37.09024,null,51.165691,47.516231,null,20.593684,23.424076,null,22.396428,52.132633,null,22.396428,-0.789275,null,51.919438,51.165691,null,45.943161,51.919438,null,38.963745,33.223191,null,52.132633,50.850346,null,22.396428,22.198745,null,45.943161,41.87194,null,35.86166,40.463667,null,35.86166,41.87194,null,22.396428,47.162494,null,35.86166,14.058324,null,51.165691,41.87194,null,22.396428,41.87194,null,37.09024,-25.274398,null,51.165691,50.850346,null,22.396428,56.130366,null,35.86166,50.850346,null,22.396428,23.685,null,52.132633,46.603354,null,22.396428,-13.254308,null,35.86166,-14.235004,null,35.86166,20.593684,null,22.396428,51.919438,null,35.86166,51.919438,null,35.86166,63.397768,null,14.058324,51.165691,null,35.86166,23.634501,null,38.963745,52.132633,null,51.165691,46.818188,null,51.165691,52.132633,null,51.165691,55.378051,null,38.963745,37.09024,null,22.396428,-25.274398,null,20.593684,55.378051,null,14.058324,36.204824,null,38.963745,51.165691,null,15.870032,51.165691,null,35.86166,61.52401,null,51.165691,49.817492,null,37.09024,51.165691,null,22.396428,-14.235004,null,35.86166,23.424076,null,51.165691,40.463667,null,53.709807,61.52401,null,35.86166,23.885942,null,22.396428,4.210484,null,22.396428,20.593684,null,51.165691,63.397768,null,22.396428,40.463667,null,-0.789275,36.204824,null,35.86166,-35.675147,null,22.396428,12.865416,null,22.396428,23.424076,null,22.396428,56.26392,null,52.132633,55.378051,null,20.593684,-0.023559,null,49.817492,51.165691,null,22.396428,-1.831239,null,35.86166,-30.559482,null,35.86166,1.352083,null,35.86166,15.870032,null,20.593684,9.081999,null,20.593684,-25.274398,null,38.963745,47.411631,null,22.396428,-18.766947,null,22.396428,50.850346,null,52.132633,40.463667,null,55.378051,53.41291,null,63.397768,60.472024,null,35.86166,56.26392,null,35.86166,33.223191,null,45.943161,51.165691,null,20.593684,30.375321,null,35.86166,-0.789275,null,52.132633,41.87194,null,35.86166,32.427908,null,20.593684,-6.369028,null,22.396428,38.963745,null,14.058324,35.907757,null,20.593684,51.165691,null,36.204824,14.058324,null,35.86166,12.879721,null,15.870032,37.09024,null,36.204824,35.86166,null,55.378051,51.165691,null,51.165691,56.26392,null,37.09024,55.378051,null,35.907757,35.86166,null,14.058324,41.87194,null,40.463667,39.399872,null,47.411631,53.709807,null,14.058324,63.397768,null,20.593684,36.204824,null,48.379433,48.669026,null,30.375321,51.165691,null,46.603354,45.943161,null,22.396428,35.907757,null,20.593684,40.463667,null,63.397768,61.92411,null,50.850346,52.132633,null,22.396428,46.818188,null,52.132633,37.09024,null,63.397768,56.26392,null,46.603354,33.886917,null,38.963745,39.074208,null,48.379433,51.165691,null,22.396428,12.862807,null,51.919438,49.817492,null,52.132633,51.919438,null,14.058324,55.378051,null,23.634501,9.748917,null,49.817492,48.669026,null,49.817492,51.919438,null,47.411631,45.943161,null,35.86166,-40.900557,null,35.86166,39.074208,null,23.885942,29.31166,null,40.463667,41.87194,null,14.058324,35.86166,null,20.593684,4.210484,null,20.593684,21.916221,null,46.603354,40.463667,null,46.603354,41.87194,null,35.907757,14.058324,null,63.397768,51.919438,null,37.09024,-38.416097,null,23.424076,29.31166,null,1.352083,35.86166,null,35.86166,4.570868,null,14.058324,40.463667,null,47.162494,51.165691,null,31.046051,12.879721,null,41.87194,46.603354,null,49.817492,46.603354,null,44.016521,45.943161,null,35.86166,46.818188,null,22.396428,63.397768,null,15.870032,36.204824,null,14.058324,51.919438,null,35.86166,31.046051,null,-0.789275,-25.274398,null,22.396428,15.870032,null,15.870032,49.817492,null,35.86166,21.916221,null,35.86166,61.92411,null,38.963745,46.603354,null,40.463667,46.603354,null,50.850346,46.603354,null,40.463667,31.791702,null,14.058324,46.603354,null,37.09024,-30.559482,null,51.165691,47.162494,null,37.09024,22.396428,null,20.593684,33.0,null,35.86166,60.472024,null,20.593684,46.603354,null,46.603354,50.850346,null,63.397768,51.165691,null,20.593684,32.427908,null,51.165691,37.09024,null,46.603354,55.378051,null,35.86166,46.151241,null,35.907757,-0.789275,null,35.86166,38.963745,null,46.603354,51.165691,null,30.375321,37.09024,null,51.165691,53.41291,null,55.169438,63.397768,null,38.963745,42.315407,null,14.058324,52.132633,null,35.86166,6.42375,null,45.943161,55.378051,null,20.593684,35.86166,null,20.593684,21.512583,null,51.165691,61.92411,null,41.87194,51.165691,null,40.463667,51.165691,null,35.86166,23.685,null,23.424076,32.427908,null,22.396428,12.879721,null,56.879635,63.397768,null,15.870032,-25.274398,null,38.963745,48.669026,null,35.86166,48.379433,null,35.86166,39.399872,null,1.352083,-25.274398,null,4.570868,23.634501,null,35.86166,26.820553,null,20.593684,56.130366,null,51.919438,48.379433,null,23.424076,23.885942,null,51.165691,26.820553,null,37.09024,-35.675147,null,22.396428,4.535277,null,20.593684,41.87194,null,46.603354,52.132633,null,37.09024,8.537981,null,39.399872,55.378051,null,51.919438,45.943161,null,35.86166,48.019573,null,22.396428,13.444304,null,22.396428,-30.559482,null,20.593684,11.825138,null,51.919438,42.733883,null,20.593684,23.885942,null,55.378051,37.09024,null,22.396428,18.735693,null,35.907757,37.09024,null,41.87194,31.046051,null,45.943161,46.603354,null,51.919438,52.132633,null,35.86166,-9.189967,null,39.399872,40.463667,null,35.86166,53.41291,null,47.411631,48.669026,null,52.132633,58.595272,null,52.132633,49.817492,null,20.593684,52.132633,null,37.09024,35.86166,null,42.733883,52.132633,null,53.41291,36.204824,null,50.850346,49.817492,null,35.907757,36.204824,null,1.352083,37.09024,null,-2.1646,1.373333,null,-0.789275,5.152149,null,51.919438,47.162494,null,51.919438,48.669026,null,22.396428,61.52401,null,14.058324,-25.274398,null,-23.442503,-14.235004,null,41.87194,38.963745,null,35.86166,-38.416097,null,20.593684,14.497401,null,35.86166,29.31166,null,20.593684,-35.675147,null,58.595272,61.92411,null,49.817492,47.516231,null,20.593684,31.046051,null,4.210484,1.352083,null,22.396428,49.817492,null,23.634501,15.783471,null,38.963745,55.378051,null,51.165691,38.963745,null,55.378051,52.132633,null,51.165691,35.86166,null,51.165691,55.169438,null,22.396428,48.669026,null,46.603354,46.818188,null,52.132633,47.516231,null,50.850346,40.463667,null,35.86166,47.516231,null,14.058324,47.516231,null,1.352083,-0.789275,null,37.09024,23.885942,null,22.396428,-29.609988,null,30.375321,55.378051,null,55.378051,1.352083,null,51.165691,45.943161,null,35.86166,47.162494,null,20.593684,-30.559482,null,38.963745,23.634501,null,42.733883,51.165691,null,37.09024,18.735693,null,52.132633,63.397768,null,23.424076,-0.023559,null,45.943161,40.463667,null,20.593684,33.223191,null,50.850346,51.165691,null,38.963745,40.463667,null,51.165691,61.52401,null,55.378051,46.603354,null,22.396428,31.046051,null,35.86166,12.862807,null,15.870032,-0.789275,null,22.396428,-40.900557,null,38.963745,63.397768,null,56.26392,60.472024,null,51.919438,46.603354,null,15.870032,52.132633,null,52.132633,49.815273,null,23.424076,21.512583,null,35.86166,9.081999,null,61.52401,48.019573,null,1.352083,4.210484,null,41.87194,39.074208,null,-25.274398,37.09024,null,35.86166,30.375321,null,45.1,51.165691,null,-0.789275,37.09024,null,51.919438,55.378051,null,41.87194,40.463667,null,20.593684,15.870032,null,51.165691,48.669026,null,51.165691,-30.559482,null,23.424076,61.92411,null,41.87194,46.818188,null,37.09024,23.424076,null,44.016521,41.87194,null,51.919438,61.52401,null,35.86166,49.817492,null,22.396428,60.472024,null,51.165691,39.399872,null,43.915886,51.165691,null,37.09024,13.794185,null,-14.235004,-9.189967,null,35.86166,31.791702,null,20.593684,13.443182,null,36.204824,50.850346,null,15.870032,23.424076,null,51.165691,60.472024,null,14.058324,61.52401,null,45.943161,48.669026,null,48.669026,46.603354,null,22.396428,32.427908,null,14.058324,56.130366,null,50.850346,55.378051,null,14.497401,-30.559482,null,35.86166,33.854721,null,51.919438,60.472024,null,1.352083,35.907757,null,-25.274398,-40.900557,null,1.352083,15.870032,null,56.26392,51.919438,null,35.86166,-6.369028,null,22.396428,47.516231,null,49.817492,52.132633,null,22.396428,46.862496,null,39.074208,26.820553,null,35.907757,1.352083,null,20.593684,-20.348404,null,-30.559482,-13.133897,null,35.86166,45.943161,null,1.352083,22.396428,null,51.165691,49.815273,null,45.943161,37.09024,null,36.204824,22.396428,null,48.379433,51.919438,null,63.397768,55.378051,null,61.52401,53.709807,null,38.963745,41.87194,null,39.399872,46.603354,null,14.058324,23.424076,null,35.86166,25.354826,null,4.210484,20.593684,null,35.86166,-0.023559,null,4.210484,61.52401,null,35.86166,12.565679,null,-1.831239,4.570868,null,51.165691,48.379433,null,38.963745,42.733883,null,46.603354,37.09024,null,35.907757,22.396428,null,30.375321,36.204824,null,38.963745,32.427908,null,55.378051,51.919438,null,-30.559482,-22.95764,null,20.593684,56.26392,null,35.86166,9.748917,null,45.943161,49.815273,null,45.943161,47.162494,null,45.943161,23.634501,null,45.943161,35.126413,null,55.378051,40.463667,null,49.817492,23.885942,null,55.378051,38.963745,null,52.132633,53.41291,null,37.09024,63.397768,null,35.86166,-18.665695,null,51.919438,55.169438,null,20.593684,7.873054,null,22.396428,36.140751,null,37.09024,4.570868,null,63.397768,50.850346,null,40.463667,52.132633,null,41.87194,37.09024,null,20.593684,-11.202692,null,35.86166,45.1,null,35.86166,30.585164,null,4.210484,38.963745,null,35.86166,7.946527,null,37.09024,20.593684,null,15.870032,63.397768,null,52.132633,46.818188,null,52.132633,56.26392,null,55.169438,58.595272,null,51.165691,23.634501,null,46.603354,39.399872,null,51.919438,23.634501,null,52.132633,23.634501,null,10.691803,6.42375,null,41.87194,45.943161,null,4.210484,41.87194,null,23.885942,25.354826,null,46.603354,51.919438,null,20.593684,-40.900557,null,35.86166,-11.202692,null,15.870032,-35.675147,null,45.943161,47.516231,null,56.26392,37.09024,null,-30.559482,-14.235004,null,37.09024,31.046051,null,51.919438,37.09024,null,35.86166,8.537981,null,31.046051,35.86166,null,31.046051,37.09024,null,55.378051,46.818188,null,37.09024,35.907757,null,37.09024,-14.235004,null,15.870032,55.378051,null,22.396428,30.375321,null,14.058324,-30.559482,null,-30.559482,-25.274398,null,38.963745,31.046051,null,36.204824,35.907757,null,35.86166,56.879635,null,4.210484,36.204824,null,47.162494,41.87194,null,51.919438,45.1,null,-35.675147,-14.235004,null,52.132633,60.472024,null,63.397768,35.86166,null,52.132633,36.204824,null,20.593684,63.397768,null,56.26392,63.397768,null,1.352083,36.204824,null,-0.789275,23.885942,null,22.396428,7.873054,null,20.593684,50.850346,null,22.396428,23.885942,null,48.379433,45.943161,null,51.165691,-14.235004,null,38.963745,48.379433,null,50.850346,45.943161,null,51.919438,63.397768,null,30.375321,-25.274398,null,35.86166,18.735693,null,46.603354,48.669026,null,51.919438,53.709807,null,41.87194,52.132633,null,49.817492,61.52401,null,55.378051,-30.559482,null,36.204824,37.09024,null,21.916221,35.907757,null,51.165691,46.151241,null,35.86166,48.669026,null,12.565679,46.603354,null,19.85627,46.603354,null,45.943161,52.132633,null,15.870032,50.850346,null,37.09024,8.619543,null,4.210484,37.09024,null,21.916221,46.603354,null,49.817492,37.09024,null,36.204824,12.879721,null,40.463667,21.521757,null,-0.789275,-40.900557,null,23.424076,20.593684,null,35.86166,-23.442503,null,38.963745,43.915886,null,51.919438,56.26392,null,-30.559482,-18.665695,null,51.165691,45.1,null,20.593684,-14.235004,null,37.09024,46.603354,null,38.963745,40.143105,null,-25.274398,4.210484,null,52.132633,42.733883,null,35.86166,28.0,null,55.378051,41.87194,null,38.963745,51.919438,null,47.162494,46.818188,null,48.669026,49.817492,null,35.86166,-0.228021,null,37.09024,15.870032,null,22.396428,30.585164,null,50.850346,53.41291,null,22.396428,15.783471,null,22.396428,33.886917,null,36.204824,-3.370417,null,23.424076,-0.228021,null,51.919438,38.963745,null,46.818188,51.165691,null,35.86166,41.377491,null,12.879721,37.09024,null,20.593684,23.685,null,31.046051,52.132633,null,37.09024,12.879721,null,41.87194,51.919438,null,-30.559482,55.378051,null,37.09024,18.971187,null,35.86166,26.3351,null,51.165691,31.791702,null,4.210484,46.603354,null,22.396428,61.92411,null,22.396428,23.634501,null,35.86166,55.169438,null,35.907757,60.472024,null,63.397768,56.130366,null,20.593684,12.879721,null,55.378051,56.26392,null,4.210484,15.870032,null,61.52401,51.165691,null,53.41291,51.165691,null,-30.559482,51.165691,null,63.397768,38.963745,null,63.397768,37.09024,null,20.593684,-17.713371,null,14.058324,23.634501,null,20.593684,1.352083,null,20.593684,60.472024,null,35.86166,58.595272,null,55.169438,61.52401,null,51.919438,40.463667,null,20.593684,35.907757,null,-18.766947,46.603354,null,37.09024,9.748917,null,51.165691,42.733883,null,51.165691,-9.189967,null,35.86166,11.825138,null,55.169438,51.919438,null,14.058324,4.210484,null,36.204824,-0.789275,null,38.963745,56.26392,null,35.907757,61.52401,null,35.86166,21.512583,null,63.397768,23.424076,null,35.86166,-1.831239,null,41.87194,49.817492,null,46.603354,31.791702,null,46.818188,6.611111,null,35.907757,21.916221,null,48.669026,47.516231,null,35.86166,42.733883,null,35.86166,9.30769,null,47.411631,61.52401,null,55.378051,61.52401,null,38.963745,45.943161,null,-30.559482,12.879721,null,47.162494,47.516231,null,-14.235004,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.5014356326588112},"lon":[104.195397,-95.712891,null,114.109497,-95.712891,null,114.109497,104.195397,null,114.109497,138.252924,null,104.195397,138.252924,null,-102.552784,-95.712891,null,121.774017,138.252924,null,78.96288,-95.712891,null,114.109497,10.451526,null,104.195397,10.451526,null,-95.712891,-106.346771,null,108.277199,-95.712891,null,10.451526,19.145136,null,104.195397,-3.435973,null,5.291266,10.451526,null,114.109497,108.277199,null,114.109497,1.888334,null,104.195397,5.291266,null,104.195397,127.766922,null,104.195397,114.109497,null,104.195397,-106.346771,null,104.195397,133.775136,null,10.451526,1.888334,null,114.109497,-3.435973,null,104.195397,1.888334,null,114.109497,103.819836,null,104.195397,101.975766,null,-70.162651,-95.712891,null,10.451526,14.550072,null,78.96288,53.847818,null,114.109497,5.291266,null,114.109497,113.921327,null,19.145136,10.451526,null,24.96676,19.145136,null,35.243322,43.679291,null,5.291266,4.351721,null,114.109497,113.543873,null,24.96676,12.56738,null,104.195397,-3.74922,null,104.195397,12.56738,null,114.109497,19.503304,null,104.195397,108.277199,null,10.451526,12.56738,null,114.109497,12.56738,null,-95.712891,133.775136,null,10.451526,4.351721,null,114.109497,-106.346771,null,104.195397,4.351721,null,114.109497,90.3563,null,5.291266,1.888334,null,114.109497,34.301525,null,104.195397,-51.92528,null,104.195397,78.96288,null,114.109497,19.145136,null,104.195397,19.145136,null,104.195397,16.354896,null,108.277199,10.451526,null,104.195397,-102.552784,null,35.243322,5.291266,null,10.451526,8.227512,null,10.451526,5.291266,null,10.451526,-3.435973,null,35.243322,-95.712891,null,114.109497,133.775136,null,78.96288,-3.435973,null,108.277199,138.252924,null,35.243322,10.451526,null,100.992541,10.451526,null,104.195397,105.318756,null,10.451526,15.472962,null,-95.712891,10.451526,null,114.109497,-51.92528,null,104.195397,53.847818,null,10.451526,-3.74922,null,27.953389,105.318756,null,104.195397,45.079162,null,114.109497,101.975766,null,114.109497,78.96288,null,10.451526,16.354896,null,114.109497,-3.74922,null,113.921327,138.252924,null,104.195397,-71.542969,null,114.109497,-85.207229,null,114.109497,53.847818,null,114.109497,9.501785,null,5.291266,-3.435973,null,78.96288,37.906193,null,15.472962,10.451526,null,114.109497,-78.183406,null,104.195397,22.937506,null,104.195397,103.819836,null,104.195397,100.992541,null,78.96288,8.675277,null,78.96288,133.775136,null,35.243322,28.369885,null,114.109497,46.869107,null,114.109497,4.351721,null,5.291266,-3.74922,null,-3.435973,-8.24389,null,16.354896,8.468946,null,104.195397,9.501785,null,104.195397,43.679291,null,24.96676,10.451526,null,78.96288,69.345116,null,104.195397,113.921327,null,5.291266,12.56738,null,104.195397,53.688046,null,78.96288,34.888822,null,114.109497,35.243322,null,108.277199,127.766922,null,78.96288,10.451526,null,138.252924,108.277199,null,104.195397,121.774017,null,100.992541,-95.712891,null,138.252924,104.195397,null,-3.435973,10.451526,null,10.451526,9.501785,null,-95.712891,-3.435973,null,127.766922,104.195397,null,108.277199,12.56738,null,-3.74922,-8.224454,null,28.369885,27.953389,null,108.277199,16.354896,null,78.96288,138.252924,null,31.16558,19.699024,null,69.345116,10.451526,null,1.888334,24.96676,null,114.109497,127.766922,null,78.96288,-3.74922,null,16.354896,25.748151,null,4.351721,5.291266,null,114.109497,8.227512,null,5.291266,-95.712891,null,16.354896,9.501785,null,1.888334,9.537499,null,35.243322,21.824312,null,31.16558,10.451526,null,114.109497,30.217636,null,19.145136,15.472962,null,5.291266,19.145136,null,108.277199,-3.435973,null,-102.552784,-83.753428,null,15.472962,19.699024,null,15.472962,19.145136,null,28.369885,24.96676,null,104.195397,174.885971,null,104.195397,21.824312,null,45.079162,47.481766,null,-3.74922,12.56738,null,108.277199,104.195397,null,78.96288,101.975766,null,78.96288,95.955974,null,1.888334,-3.74922,null,1.888334,12.56738,null,127.766922,108.277199,null,16.354896,19.145136,null,-95.712891,-63.616672,null,53.847818,47.481766,null,103.819836,104.195397,null,104.195397,-74.297333,null,108.277199,-3.74922,null,19.503304,10.451526,null,34.851612,121.774017,null,12.56738,1.888334,null,15.472962,1.888334,null,21.005859,24.96676,null,104.195397,8.227512,null,114.109497,16.354896,null,100.992541,138.252924,null,108.277199,19.145136,null,104.195397,34.851612,null,113.921327,133.775136,null,114.109497,100.992541,null,100.992541,15.472962,null,104.195397,95.955974,null,104.195397,25.748151,null,35.243322,1.888334,null,-3.74922,1.888334,null,4.351721,1.888334,null,-3.74922,-7.09262,null,108.277199,1.888334,null,-95.712891,22.937506,null,10.451526,19.503304,null,-95.712891,114.109497,null,78.96288,65.0,null,104.195397,8.468946,null,78.96288,1.888334,null,1.888334,4.351721,null,16.354896,10.451526,null,78.96288,53.688046,null,10.451526,-95.712891,null,1.888334,-3.435973,null,104.195397,14.995463,null,127.766922,113.921327,null,104.195397,35.243322,null,1.888334,10.451526,null,69.345116,-95.712891,null,10.451526,-8.24389,null,23.881275,16.354896,null,35.243322,43.356892,null,108.277199,5.291266,null,104.195397,-66.58973,null,24.96676,-3.435973,null,78.96288,104.195397,null,78.96288,55.923255,null,10.451526,25.748151,null,12.56738,10.451526,null,-3.74922,10.451526,null,104.195397,90.3563,null,53.847818,53.688046,null,114.109497,121.774017,null,24.603189,16.354896,null,100.992541,133.775136,null,35.243322,19.699024,null,104.195397,31.16558,null,104.195397,-8.224454,null,103.819836,133.775136,null,-74.297333,-102.552784,null,104.195397,30.802498,null,78.96288,-106.346771,null,19.145136,31.16558,null,53.847818,45.079162,null,10.451526,30.802498,null,-95.712891,-71.542969,null,114.109497,114.727669,null,78.96288,12.56738,null,1.888334,5.291266,null,-95.712891,-80.782127,null,-8.224454,-3.435973,null,19.145136,24.96676,null,104.195397,66.923684,null,114.109497,144.793731,null,114.109497,22.937506,null,78.96288,42.590275,null,19.145136,25.48583,null,78.96288,45.079162,null,-3.435973,-95.712891,null,114.109497,-70.162651,null,127.766922,-95.712891,null,12.56738,34.851612,null,24.96676,1.888334,null,19.145136,5.291266,null,104.195397,-75.015152,null,-8.224454,-3.74922,null,104.195397,-8.24389,null,28.369885,19.699024,null,5.291266,25.013607,null,5.291266,15.472962,null,78.96288,5.291266,null,-95.712891,104.195397,null,25.48583,5.291266,null,-8.24389,138.252924,null,4.351721,15.472962,null,127.766922,138.252924,null,103.819836,-95.712891,null,24.15536,32.290275,null,113.921327,46.199616,null,19.145136,19.503304,null,19.145136,19.699024,null,114.109497,105.318756,null,108.277199,133.775136,null,-58.443832,-51.92528,null,12.56738,35.243322,null,104.195397,-63.616672,null,78.96288,-14.452362,null,104.195397,47.481766,null,78.96288,-71.542969,null,25.013607,25.748151,null,15.472962,14.550072,null,78.96288,34.851612,null,101.975766,103.819836,null,114.109497,15.472962,null,-102.552784,-90.230759,null,35.243322,-3.435973,null,10.451526,35.243322,null,-3.435973,5.291266,null,10.451526,104.195397,null,10.451526,23.881275,null,114.109497,19.699024,null,1.888334,8.227512,null,5.291266,14.550072,null,4.351721,-3.74922,null,104.195397,14.550072,null,108.277199,14.550072,null,103.819836,113.921327,null,-95.712891,45.079162,null,114.109497,28.233608,null,69.345116,-3.435973,null,-3.435973,103.819836,null,10.451526,24.96676,null,104.195397,19.503304,null,78.96288,22.937506,null,35.243322,-102.552784,null,25.48583,10.451526,null,-95.712891,-70.162651,null,5.291266,16.354896,null,53.847818,37.906193,null,24.96676,-3.74922,null,78.96288,43.679291,null,4.351721,10.451526,null,35.243322,-3.74922,null,10.451526,105.318756,null,-3.435973,1.888334,null,114.109497,34.851612,null,104.195397,30.217636,null,100.992541,113.921327,null,114.109497,174.885971,null,35.243322,16.354896,null,9.501785,8.468946,null,19.145136,1.888334,null,100.992541,5.291266,null,5.291266,6.129583,null,53.847818,55.923255,null,104.195397,8.675277,null,105.318756,66.923684,null,103.819836,101.975766,null,12.56738,21.824312,null,133.775136,-95.712891,null,104.195397,69.345116,null,15.2,10.451526,null,113.921327,-95.712891,null,19.145136,-3.435973,null,12.56738,-3.74922,null,78.96288,100.992541,null,10.451526,19.699024,null,10.451526,22.937506,null,53.847818,25.748151,null,12.56738,8.227512,null,-95.712891,53.847818,null,21.005859,12.56738,null,19.145136,105.318756,null,104.195397,15.472962,null,114.109497,8.468946,null,10.451526,-8.224454,null,17.679076,10.451526,null,-95.712891,-88.89653,null,-51.92528,-75.015152,null,104.195397,-7.09262,null,78.96288,-15.310139,null,138.252924,4.351721,null,100.992541,53.847818,null,10.451526,8.468946,null,108.277199,105.318756,null,24.96676,19.699024,null,19.699024,1.888334,null,114.109497,53.688046,null,108.277199,-106.346771,null,4.351721,-3.435973,null,-14.452362,22.937506,null,104.195397,35.862285,null,19.145136,8.468946,null,103.819836,127.766922,null,133.775136,174.885971,null,103.819836,100.992541,null,9.501785,19.145136,null,104.195397,34.888822,null,114.109497,14.550072,null,15.472962,5.291266,null,114.109497,103.846656,null,21.824312,30.802498,null,127.766922,103.819836,null,78.96288,57.552152,null,22.937506,27.849332,null,104.195397,24.96676,null,103.819836,114.109497,null,10.451526,6.129583,null,24.96676,-95.712891,null,138.252924,114.109497,null,31.16558,19.145136,null,16.354896,-3.435973,null,105.318756,27.953389,null,35.243322,12.56738,null,-8.224454,1.888334,null,108.277199,53.847818,null,104.195397,51.183884,null,101.975766,78.96288,null,104.195397,37.906193,null,101.975766,105.318756,null,104.195397,104.990963,null,-78.183406,-74.297333,null,10.451526,31.16558,null,35.243322,25.48583,null,1.888334,-95.712891,null,127.766922,114.109497,null,69.345116,138.252924,null,35.243322,53.688046,null,-3.435973,19.145136,null,22.937506,18.49041,null,78.96288,9.501785,null,104.195397,-83.753428,null,24.96676,6.129583,null,24.96676,19.503304,null,24.96676,-102.552784,null,24.96676,33.429859,null,-3.435973,-3.74922,null,15.472962,45.079162,null,-3.435973,35.243322,null,5.291266,-8.24389,null,-95.712891,16.354896,null,104.195397,35.529562,null,19.145136,23.881275,null,78.96288,80.771797,null,114.109497,-5.353585,null,-95.712891,-74.297333,null,16.354896,4.351721,null,-3.74922,5.291266,null,12.56738,-95.712891,null,78.96288,17.873887,null,104.195397,15.2,null,104.195397,36.238414,null,101.975766,35.243322,null,104.195397,-1.023194,null,-95.712891,78.96288,null,100.992541,16.354896,null,5.291266,8.227512,null,5.291266,9.501785,null,23.881275,25.013607,null,10.451526,-102.552784,null,1.888334,-8.224454,null,19.145136,-102.552784,null,5.291266,-102.552784,null,-61.222503,-66.58973,null,12.56738,24.96676,null,101.975766,12.56738,null,45.079162,51.183884,null,1.888334,19.145136,null,78.96288,174.885971,null,104.195397,17.873887,null,100.992541,-71.542969,null,24.96676,14.550072,null,9.501785,-95.712891,null,22.937506,-51.92528,null,-95.712891,34.851612,null,19.145136,-95.712891,null,104.195397,-80.782127,null,34.851612,104.195397,null,34.851612,-95.712891,null,-3.435973,8.227512,null,-95.712891,127.766922,null,-95.712891,-51.92528,null,100.992541,-3.435973,null,114.109497,69.345116,null,108.277199,22.937506,null,22.937506,133.775136,null,35.243322,34.851612,null,138.252924,127.766922,null,104.195397,24.603189,null,101.975766,138.252924,null,19.503304,12.56738,null,19.145136,15.2,null,-71.542969,-51.92528,null,5.291266,8.468946,null,16.354896,104.195397,null,5.291266,138.252924,null,78.96288,16.354896,null,9.501785,16.354896,null,103.819836,138.252924,null,113.921327,45.079162,null,114.109497,80.771797,null,78.96288,4.351721,null,114.109497,45.079162,null,31.16558,24.96676,null,10.451526,-51.92528,null,35.243322,31.16558,null,4.351721,24.96676,null,19.145136,16.354896,null,69.345116,133.775136,null,104.195397,-70.162651,null,1.888334,19.699024,null,19.145136,27.953389,null,12.56738,5.291266,null,15.472962,105.318756,null,-3.435973,22.937506,null,138.252924,-95.712891,null,95.955974,127.766922,null,10.451526,14.995463,null,104.195397,19.699024,null,104.990963,1.888334,null,102.495496,1.888334,null,24.96676,5.291266,null,100.992541,4.351721,null,-95.712891,0.824782,null,101.975766,-95.712891,null,95.955974,1.888334,null,15.472962,-95.712891,null,138.252924,121.774017,null,-3.74922,-77.781167,null,113.921327,174.885971,null,53.847818,78.96288,null,104.195397,-58.443832,null,35.243322,17.679076,null,19.145136,9.501785,null,22.937506,35.529562,null,10.451526,15.2,null,78.96288,-51.92528,null,-95.712891,1.888334,null,35.243322,47.576927,null,133.775136,101.975766,null,5.291266,25.48583,null,104.195397,3.0,null,-3.435973,12.56738,null,35.243322,19.145136,null,19.503304,8.227512,null,19.699024,15.472962,null,104.195397,15.827659,null,-95.712891,100.992541,null,114.109497,36.238414,null,4.351721,-8.24389,null,114.109497,-90.230759,null,114.109497,9.537499,null,138.252924,-168.734039,null,53.847818,15.827659,null,19.145136,35.243322,null,8.227512,10.451526,null,104.195397,64.585262,null,121.774017,-95.712891,null,78.96288,90.3563,null,34.851612,5.291266,null,-95.712891,121.774017,null,12.56738,19.145136,null,22.937506,-3.435973,null,-95.712891,-72.285215,null,104.195397,17.228331,null,10.451526,-7.09262,null,101.975766,1.888334,null,114.109497,25.748151,null,114.109497,-102.552784,null,104.195397,23.881275,null,127.766922,8.468946,null,16.354896,-106.346771,null,78.96288,121.774017,null,-3.435973,9.501785,null,101.975766,100.992541,null,105.318756,10.451526,null,-8.24389,10.451526,null,22.937506,10.451526,null,16.354896,35.243322,null,16.354896,-95.712891,null,78.96288,178.065033,null,108.277199,-102.552784,null,78.96288,103.819836,null,78.96288,8.468946,null,104.195397,25.013607,null,23.881275,105.318756,null,19.145136,-3.74922,null,78.96288,127.766922,null,46.869107,1.888334,null,-95.712891,-83.753428,null,10.451526,25.48583,null,10.451526,-75.015152,null,104.195397,42.590275,null,23.881275,19.145136,null,108.277199,101.975766,null,138.252924,113.921327,null,35.243322,9.501785,null,127.766922,105.318756,null,104.195397,55.923255,null,16.354896,53.847818,null,104.195397,-78.183406,null,12.56738,15.472962,null,1.888334,-7.09262,null,8.227512,20.939444,null,127.766922,95.955974,null,19.699024,14.550072,null,104.195397,25.48583,null,104.195397,2.315834,null,28.369885,105.318756,null,-3.435973,105.318756,null,35.243322,24.96676,null,22.937506,121.774017,null,19.503304,14.550072,null,-51.92528,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[-11.202692,50.850346,7.946527,39.399872,17.060816,15.414999,-38.416097,-25.274398,-16.290154,-14.235004,-35.675147,4.570868,9.748917,18.735693,15.783471,-23.442503,-9.189967,51.919438,40.463667,-32.522779,4.535277,56.130366,51.165691,22.396428,47.162494,-0.789275,41.87194,36.204824,52.132633,-20.904305,-40.900557,-6.314993,12.879721,1.352083,-30.559482,15.870032,26.820553,55.378051,-6.369028,37.09024,-13.759029,26.02751,23.885942,23.424076,33.0,42.546245,40.143105,47.516231,43.915886,42.733883,7.369722,35.86166,-0.228021,-2.1646,45.1,21.521757,35.126413,49.817492,9.30769,56.26392,1.650801,58.595272,61.92411,46.603354,36.140751,39.074208,9.945587,64.963051,33.223191,53.41291,31.046051,48.019573,35.907757,29.31166,33.854721,56.879635,55.169438,49.815273,4.210484,35.937496,23.634501,47.411631,31.791702,-18.665695,60.472024,25.354826,45.943161,61.52401,-1.940278,14.497401,44.016521,-4.679574,48.669026,46.151241,12.862807,63.397768,46.818188,8.619543,33.886917,38.963745,48.379433,41.608635,19.3133,-1.831239,4.860416,15.199999,8.537981,20.593684,40.069099,42.315407,17.570692,21.916221,53.709807,38.969719,12.565679,6.428055,46.946947,14.058324,10.691803,13.794185,12.865416,41.0,28.0,25.03428,23.685,13.193887,-22.328474,17.189877,-9.64571,-3.373056,16.5388,7.873054,15.454166,-11.6455,9.145,-17.713371,-17.679742,11.825138,-0.803689,13.443182,12.262776,18.971187,32.427908,7.539989,18.109581,30.585164,-0.023559,40.339852,41.20438,-29.609988,26.3351,22.198745,-18.766947,-13.254308,3.202778,21.00789,-20.348404,46.862496,42.708678,21.512583,-22.95764,28.394857,12.16957,12.52111,-15.376706,17.607789,9.081999,30.375321,-8.874217,13.909444,8.460555,5.152149,-19.015438,7.862684,3.919305,34.802075,38.861034,21.694025,1.373333,41.377491,6.42375,15.552727,-13.133897,61.892635,71.706936,-21.178986,6.611111,18.04248,17.897476,12.238333,43.94236,19.85627,7.51498,13.444304,17.664332,17.357822,32.3078,11.803749,0.18636,-90.0,-26.522503,12.984305,18.420695,-14.28522,-10.447525,-0.522778,15.179384,31.952162,-3.370417,-7.109535,27.514162,-13.768752,-15.965,7.131474,-12.164165,12.20189,-51.796253,-21.236736,7.425554,16.742498,-49.280366,41.902916,24.215527,-19.054445,-29.040835,-9.2,18.218785],"lon":[17.873887,4.351721,-1.023194,-8.224454,-61.796428,-61.370976,-63.616672,133.775136,-63.588653,-51.92528,-71.542969,-74.297333,-83.753428,-70.162651,-90.230759,-58.443832,-75.015152,19.145136,-3.74922,-55.765835,114.727669,-106.346771,10.451526,114.109497,19.503304,113.921327,12.56738,138.252924,5.291266,165.618042,174.885971,143.95555,121.774017,103.819836,22.937506,100.992541,30.802498,-3.435973,34.888822,-95.712891,-172.104629,50.55096,45.079162,53.847818,65.0,1.601554,47.576927,14.550072,17.679076,25.48583,12.354722,104.195397,15.827659,24.15536,15.2,-77.781167,33.429859,15.472962,2.315834,9.501785,10.267895,25.013607,25.748151,1.888334,-5.353585,21.824312,-9.696645,-19.020835,43.679291,-8.24389,34.851612,66.923684,127.766922,47.481766,35.862285,24.603189,23.881275,6.129583,101.975766,14.375416,-102.552784,28.369885,-7.09262,35.529562,8.468946,51.183884,24.96676,105.318756,29.873888,-14.452362,21.005859,55.491977,19.699024,14.995463,30.217636,16.354896,8.227512,0.824782,9.537499,35.243322,31.16558,21.745275,-81.2546,-78.183406,-58.93018,-86.241905,-80.782127,78.96288,45.038189,43.356892,-3.996166,95.955974,27.953389,59.556278,104.990963,-9.429499,-56.32509,108.277199,-61.222503,-88.89653,-85.207229,20.0,3.0,-77.39628,90.3563,-59.543198,24.684866,-88.49765,160.156194,29.918886,-23.0418,80.771797,18.732207,43.3333,40.489673,178.065033,-149.406843,42.590275,11.609444,-15.310139,-61.604171,-72.285215,53.688046,-5.54708,-77.297508,36.238414,37.906193,127.510093,74.766098,28.233608,17.228331,113.543873,46.869107,34.301525,73.22068,-10.940835,57.552152,103.846656,19.37439,55.923255,18.49041,84.124008,-68.990021,-69.968338,166.959158,8.081666,8.675277,69.345116,125.727,-60.978893,-11.779889,46.199616,29.154857,30.217636,-56.027783,38.996815,71.276093,-71.797928,32.290275,64.585262,-66.58973,48.516388,27.849332,-6.911806,-42.604303,-175.198242,20.939444,-63.05483,-62.83055,-1.561593,12.457777,102.495496,134.58252,144.793731,145.94351,-62.782998,-64.7505,-15.180413,6.613081,0.0,31.465866,-61.287228,-64.639968,-170.70444,105.690449,166.931503,39.782334,35.233154,-168.734039,179.194167,90.433601,-177.156097,-5.7089,171.184478,96.870956,-68.262383,-59.523613,-159.777671,150.550812,-62.187366,69.348557,12.453389,-12.885834,-169.867233,167.954712,-171.833333,-63.043653],"marker":{"color":"red","opacity":0.8,"size":[18.0,20.0,16.8,20.0,8.4,3.2,20.0,20.0,11.6,20.0,20.0,20.0,13.2,20.0,18.0,11.2,20.0,20.0,20.0,13.2,6.8,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,7.2,20.0,6.4,20.0,20.0,20.0,20.0,17.2,20.0,9.6,20.0,2.4,15.6,20.0,20.0,7.2,6.4,15.2,20.0,20.0,20.0,8.8,20.0,10.8,10.4,20.0,8.0,18.0,20.0,4.8,20.0,4.8,20.0,20.0,20.0,6.8,20.0,4.8,16.0,10.4,20.0,20.0,20.0,20.0,20.0,15.2,20.0,20.0,20.0,20.0,12.4,20.0,20.0,14.0,8.0,20.0,15.6,20.0,20.0,8.0,12.8,20.0,8.8,20.0,20.0,8.0,20.0,20.0,5.6,10.4,20.0,20.0,15.2,5.6,18.8,8.4,14.4,19.6,20.0,12.0,19.6,8.0,20.0,20.0,6.4,9.6,4.8,1.2,20.0,20.0,14.4,9.6,11.6,11.2,7.6,9.6,13.2,6.8,6.4,3.6,2.0,3.6,10.8,3.6,2.0,10.8,13.6,9.2,5.2,8.0,4.4,4.4,5.2,12.0,13.6,10.4,15.2,19.6,1.2,15.2,2.4,8.8,11.6,14.4,4.0,8.8,7.2,19.2,16.4,9.6,13.2,17.6,5.2,7.6,7.2,4.0,3.2,11.2,20.0,2.8,4.4,6.0,5.2,6.0,2.8,6.8,4.0,6.4,2.4,10.4,7.2,9.2,4.8,7.6,4.0,5.2,2.0,4.4,4.0,0.8,7.2,2.8,5.6,2.8,4.4,2.4,3.6,7.2,2.4,2.0,1.2,5.6,3.6,4.0,1.2,0.4,1.6,3.2,2.8,2.4,0.8,3.2,1.2,0.4,2.8,0.4,1.2,0.4,2.0,1.6,0.4,0.8,0.8,0.4,0.4,0.4,0.4,0.4]},"mode":"markers","name":"2017","text":["AGO\u003cbr\u003eConex\u00f5es: 45","BEL\u003cbr\u003eConex\u00f5es: 213","GHA\u003cbr\u003eConex\u00f5es: 42","PRT\u003cbr\u003eConex\u00f5es: 165","ATG\u003cbr\u003eConex\u00f5es: 21","DMA\u003cbr\u003eConex\u00f5es: 8","ARG\u003cbr\u003eConex\u00f5es: 68","AUS\u003cbr\u003eConex\u00f5es: 139","BOL\u003cbr\u003eConex\u00f5es: 29","BRA\u003cbr\u003eConex\u00f5es: 140","CHL\u003cbr\u003eConex\u00f5es: 64","COL\u003cbr\u003eConex\u00f5es: 85","CRI\u003cbr\u003eConex\u00f5es: 33","DOM\u003cbr\u003eConex\u00f5es: 59","GTM\u003cbr\u003eConex\u00f5es: 45","PRY\u003cbr\u003eConex\u00f5es: 28","PER\u003cbr\u003eConex\u00f5es: 83","POL\u003cbr\u003eConex\u00f5es: 186","ESP\u003cbr\u003eConex\u00f5es: 198","URY\u003cbr\u003eConex\u00f5es: 33","BRN\u003cbr\u003eConex\u00f5es: 17","CAN\u003cbr\u003eConex\u00f5es: 141","DEU\u003cbr\u003eConex\u00f5es: 250","HKG\u003cbr\u003eConex\u00f5es: 178","HUN\u003cbr\u003eConex\u00f5es: 123","IDN\u003cbr\u003eConex\u00f5es: 111","ITA\u003cbr\u003eConex\u00f5es: 197","JPN\u003cbr\u003eConex\u00f5es: 143","NLD\u003cbr\u003eConex\u00f5es: 225","NCL\u003cbr\u003eConex\u00f5es: 18","NZL\u003cbr\u003eConex\u00f5es: 88","PNG\u003cbr\u003eConex\u00f5es: 16","PHL\u003cbr\u003eConex\u00f5es: 61","SGP\u003cbr\u003eConex\u00f5es: 141","ZAF\u003cbr\u003eConex\u00f5es: 158","THA\u003cbr\u003eConex\u00f5es: 160","EGY\u003cbr\u003eConex\u00f5es: 43","GBR\u003cbr\u003eConex\u00f5es: 216","TZA\u003cbr\u003eConex\u00f5es: 24","USA\u003cbr\u003eConex\u00f5es: 228","WSM\u003cbr\u003eConex\u00f5es: 6","BHR\u003cbr\u003eConex\u00f5es: 39","SAU\u003cbr\u003eConex\u00f5es: 52","ARE\u003cbr\u003eConex\u00f5es: 184","AFG\u003cbr\u003eConex\u00f5es: 18","AND\u003cbr\u003eConex\u00f5es: 16","AZE\u003cbr\u003eConex\u00f5es: 38","AUT\u003cbr\u003eConex\u00f5es: 54","BIH\u003cbr\u003eConex\u00f5es: 57","BGR\u003cbr\u003eConex\u00f5es: 89","CMR\u003cbr\u003eConex\u00f5es: 22","CHN\u003cbr\u003eConex\u00f5es: 268","COG\u003cbr\u003eConex\u00f5es: 27","COD\u003cbr\u003eConex\u00f5es: 26","HRV\u003cbr\u003eConex\u00f5es: 87","CUB\u003cbr\u003eConex\u00f5es: 20","CYP\u003cbr\u003eConex\u00f5es: 45","CZE\u003cbr\u003eConex\u00f5es: 141","BEN\u003cbr\u003eConex\u00f5es: 12","DNK\u003cbr\u003eConex\u00f5es: 155","GNQ\u003cbr\u003eConex\u00f5es: 12","EST\u003cbr\u003eConex\u00f5es: 80","FIN\u003cbr\u003eConex\u00f5es: 119","FRA\u003cbr\u003eConex\u00f5es: 252","GIB\u003cbr\u003eConex\u00f5es: 17","GRC\u003cbr\u003eConex\u00f5es: 102","GIN\u003cbr\u003eConex\u00f5es: 12","ISL\u003cbr\u003eConex\u00f5es: 40","IRQ\u003cbr\u003eConex\u00f5es: 26","IRL\u003cbr\u003eConex\u00f5es: 96","ISR\u003cbr\u003eConex\u00f5es: 91","KAZ\u003cbr\u003eConex\u00f5es: 71","KOR\u003cbr\u003eConex\u00f5es: 171","KWT\u003cbr\u003eConex\u00f5es: 51","LBN\u003cbr\u003eConex\u00f5es: 38","LVA\u003cbr\u003eConex\u00f5es: 103","LTU\u003cbr\u003eConex\u00f5es: 115","LUX\u003cbr\u003eConex\u00f5es: 75","MYS\u003cbr\u003eConex\u00f5es: 93","MLT\u003cbr\u003eConex\u00f5es: 31","MEX\u003cbr\u003eConex\u00f5es: 76","MDA\u003cbr\u003eConex\u00f5es: 50","MAR\u003cbr\u003eConex\u00f5es: 35","MOZ\u003cbr\u003eConex\u00f5es: 20","NOR\u003cbr\u003eConex\u00f5es: 115","QAT\u003cbr\u003eConex\u00f5es: 39","ROU\u003cbr\u003eConex\u00f5es: 118","RUS\u003cbr\u003eConex\u00f5es: 141","RWA\u003cbr\u003eConex\u00f5es: 20","SEN\u003cbr\u003eConex\u00f5es: 32","SRB\u003cbr\u003eConex\u00f5es: 93","SYC\u003cbr\u003eConex\u00f5es: 22","SVK\u003cbr\u003eConex\u00f5es: 116","SVN\u003cbr\u003eConex\u00f5es: 110","SDN\u003cbr\u003eConex\u00f5es: 20","SWE\u003cbr\u003eConex\u00f5es: 173","CHE\u003cbr\u003eConex\u00f5es: 200","TGO\u003cbr\u003eConex\u00f5es: 14","TUN\u003cbr\u003eConex\u00f5es: 26","TUR\u003cbr\u003eConex\u00f5es: 183","UKR\u003cbr\u003eConex\u00f5es: 112","MKD\u003cbr\u003eConex\u00f5es: 38","CYM\u003cbr\u003eConex\u00f5es: 14","ECU\u003cbr\u003eConex\u00f5es: 47","GUY\u003cbr\u003eConex\u00f5es: 21","HND\u003cbr\u003eConex\u00f5es: 36","PAN\u003cbr\u003eConex\u00f5es: 49","IND\u003cbr\u003eConex\u00f5es: 222","ARM\u003cbr\u003eConex\u00f5es: 30","GEO\u003cbr\u003eConex\u00f5es: 49","MLI\u003cbr\u003eConex\u00f5es: 20","MMR\u003cbr\u003eConex\u00f5es: 57","BLR\u003cbr\u003eConex\u00f5es: 59","TKM\u003cbr\u003eConex\u00f5es: 16","KHM\u003cbr\u003eConex\u00f5es: 24","LBR\u003cbr\u003eConex\u00f5es: 12","SPM\u003cbr\u003eConex\u00f5es: 3","VNM\u003cbr\u003eConex\u00f5es: 99","TTO\u003cbr\u003eConex\u00f5es: 56","SLV\u003cbr\u003eConex\u00f5es: 36","NIC\u003cbr\u003eConex\u00f5es: 24","ALB\u003cbr\u003eConex\u00f5es: 29","DZA\u003cbr\u003eConex\u00f5es: 28","BHS\u003cbr\u003eConex\u00f5es: 19","BGD\u003cbr\u003eConex\u00f5es: 24","BRB\u003cbr\u003eConex\u00f5es: 33","BWA\u003cbr\u003eConex\u00f5es: 17","BLZ\u003cbr\u003eConex\u00f5es: 16","SLB\u003cbr\u003eConex\u00f5es: 9","BDI\u003cbr\u003eConex\u00f5es: 5","CPV\u003cbr\u003eConex\u00f5es: 9","LKA\u003cbr\u003eConex\u00f5es: 27","TCD\u003cbr\u003eConex\u00f5es: 9","COM\u003cbr\u003eConex\u00f5es: 5","ETH\u003cbr\u003eConex\u00f5es: 27","FJI\u003cbr\u003eConex\u00f5es: 34","PYF\u003cbr\u003eConex\u00f5es: 23","DJI\u003cbr\u003eConex\u00f5es: 13","GAB\u003cbr\u003eConex\u00f5es: 20","GMB\u003cbr\u003eConex\u00f5es: 11","GRD\u003cbr\u003eConex\u00f5es: 11","HTI\u003cbr\u003eConex\u00f5es: 13","IRN\u003cbr\u003eConex\u00f5es: 30","CIV\u003cbr\u003eConex\u00f5es: 34","JAM\u003cbr\u003eConex\u00f5es: 26","JOR\u003cbr\u003eConex\u00f5es: 38","KEN\u003cbr\u003eConex\u00f5es: 49","PRK\u003cbr\u003eConex\u00f5es: 3","KGZ\u003cbr\u003eConex\u00f5es: 38","LSO\u003cbr\u003eConex\u00f5es: 6","LBY\u003cbr\u003eConex\u00f5es: 22","MAC\u003cbr\u003eConex\u00f5es: 29","MDG\u003cbr\u003eConex\u00f5es: 36","MWI\u003cbr\u003eConex\u00f5es: 10","MDV\u003cbr\u003eConex\u00f5es: 22","MRT\u003cbr\u003eConex\u00f5es: 18","MUS\u003cbr\u003eConex\u00f5es: 48","MNG\u003cbr\u003eConex\u00f5es: 41","MNE\u003cbr\u003eConex\u00f5es: 24","OMN\u003cbr\u003eConex\u00f5es: 33","NAM\u003cbr\u003eConex\u00f5es: 44","NPL\u003cbr\u003eConex\u00f5es: 13","CUW\u003cbr\u003eConex\u00f5es: 19","ABW\u003cbr\u003eConex\u00f5es: 18","VUT\u003cbr\u003eConex\u00f5es: 10","NER\u003cbr\u003eConex\u00f5es: 8","NGA\u003cbr\u003eConex\u00f5es: 28","PAK\u003cbr\u003eConex\u00f5es: 144","TLS\u003cbr\u003eConex\u00f5es: 7","LCA\u003cbr\u003eConex\u00f5es: 11","SLE\u003cbr\u003eConex\u00f5es: 15","SOM\u003cbr\u003eConex\u00f5es: 13","ZWE\u003cbr\u003eConex\u00f5es: 15","SSD\u003cbr\u003eConex\u00f5es: 7","SUR\u003cbr\u003eConex\u00f5es: 17","SYR\u003cbr\u003eConex\u00f5es: 10","TJK\u003cbr\u003eConex\u00f5es: 16","TCA\u003cbr\u003eConex\u00f5es: 6","UGA\u003cbr\u003eConex\u00f5es: 26","UZB\u003cbr\u003eConex\u00f5es: 18","VEN\u003cbr\u003eConex\u00f5es: 23","YEM\u003cbr\u003eConex\u00f5es: 12","ZMB\u003cbr\u003eConex\u00f5es: 19","FRO\u003cbr\u003eConex\u00f5es: 10","GRL\u003cbr\u003eConex\u00f5es: 13","TON\u003cbr\u003eConex\u00f5es: 5","CAF\u003cbr\u003eConex\u00f5es: 11","SXM\u003cbr\u003eConex\u00f5es: 10","BLM\u003cbr\u003eConex\u00f5es: 2","BFA\u003cbr\u003eConex\u00f5es: 18","SMR\u003cbr\u003eConex\u00f5es: 7","LAO\u003cbr\u003eConex\u00f5es: 14","PLW\u003cbr\u003eConex\u00f5es: 7","GUM\u003cbr\u003eConex\u00f5es: 11","MNP\u003cbr\u003eConex\u00f5es: 6","KNA\u003cbr\u003eConex\u00f5es: 9","BMU\u003cbr\u003eConex\u00f5es: 18","GNB\u003cbr\u003eConex\u00f5es: 6","STP\u003cbr\u003eConex\u00f5es: 5","ATA\u003cbr\u003eConex\u00f5es: 3","SWZ\u003cbr\u003eConex\u00f5es: 14","VCT\u003cbr\u003eConex\u00f5es: 9","VGB\u003cbr\u003eConex\u00f5es: 10","ASM\u003cbr\u003eConex\u00f5es: 3","CXR\u003cbr\u003eConex\u00f5es: 1","NRU\u003cbr\u003eConex\u00f5es: 4","ERI\u003cbr\u003eConex\u00f5es: 8","PSE\u003cbr\u003eConex\u00f5es: 7","KIR\u003cbr\u003eConex\u00f5es: 6","TUV\u003cbr\u003eConex\u00f5es: 2","BTN\u003cbr\u003eConex\u00f5es: 8","WLF\u003cbr\u003eConex\u00f5es: 3","SHN\u003cbr\u003eConex\u00f5es: 1","MHL\u003cbr\u003eConex\u00f5es: 7","CCK\u003cbr\u003eConex\u00f5es: 1","BES\u003cbr\u003eConex\u00f5es: 3","FLK\u003cbr\u003eConex\u00f5es: 1","COK\u003cbr\u003eConex\u00f5es: 5","FSM\u003cbr\u003eConex\u00f5es: 4","MSR\u003cbr\u003eConex\u00f5es: 1","ATF\u003cbr\u003eConex\u00f5es: 2","VAT\u003cbr\u003eConex\u00f5es: 2","ESH\u003cbr\u003eConex\u00f5es: 1","NIU\u003cbr\u003eConex\u00f5es: 1","NFK\u003cbr\u003eConex\u00f5es: 1","TKL\u003cbr\u003eConex\u00f5es: 1","AIA\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.82041091},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,36.204824,null,35.86166,51.165691,null,35.86166,55.378051,null,35.86166,52.132633,null,23.634501,37.09024,null,35.86166,56.130366,null,37.09024,56.130366,null,35.86166,46.603354,null,35.86166,-25.274398,null,35.86166,35.907757,null,14.058324,37.09024,null,35.86166,40.463667,null,52.132633,51.165691,null,35.86166,41.87194,null,35.86166,51.919438,null,35.86166,50.850346,null,35.86166,61.52401,null,35.86166,4.210484,null,20.593684,37.09024,null,35.86166,20.593684,null,35.86166,23.885942,null,35.86166,-14.235004,null,35.86166,63.397768,null,51.165691,46.603354,null,35.86166,23.634501,null,38.963745,37.09024,null,35.86166,-35.675147,null,35.86166,22.396428,null,35.86166,33.223191,null,38.963745,52.132633,null,35.86166,14.058324,null,51.165691,51.919438,null,18.735693,37.09024,null,20.593684,23.424076,null,52.132633,50.850346,null,35.86166,23.424076,null,12.879721,36.204824,null,38.963745,51.165691,null,52.132633,46.603354,null,35.86166,12.879721,null,51.165691,47.516231,null,31.791702,40.463667,null,45.943161,51.919438,null,51.919438,51.165691,null,51.165691,52.132633,null,35.86166,15.870032,null,35.86166,-0.789275,null,35.86166,1.352083,null,37.09024,50.850346,null,35.86166,56.26392,null,51.165691,41.87194,null,35.86166,61.92411,null,37.09024,-25.274398,null,35.86166,-40.900557,null,51.165691,49.817492,null,14.058324,51.165691,null,14.058324,36.204824,null,35.86166,46.151241,null,15.870032,37.09024,null,35.86166,-30.559482,null,15.870032,51.165691,null,51.165691,46.818188,null,35.86166,9.081999,null,22.396428,35.86166,null,37.09024,51.165691,null,51.165691,55.378051,null,38.963745,26.3351,null,35.86166,-9.189967,null,35.86166,53.41291,null,35.86166,4.570868,null,35.86166,39.074208,null,51.165691,40.463667,null,46.603354,28.0,null,50.850346,52.132633,null,35.86166,21.916221,null,35.86166,31.046051,null,35.86166,38.963745,null,49.817492,51.165691,null,35.86166,60.472024,null,35.86166,26.820553,null,38.963745,47.411631,null,20.593684,33.0,null,55.378051,51.165691,null,20.593684,9.081999,null,55.378051,53.41291,null,22.396428,37.09024,null,40.463667,46.603354,null,35.86166,48.379433,null,35.86166,-0.228021,null,40.463667,39.399872,null,51.919438,49.817492,null,51.165691,63.397768,null,35.86166,49.817492,null,-14.235004,-23.442503,null,20.593684,23.685,null,51.165691,50.850346,null,35.86166,30.375321,null,52.132633,55.378051,null,35.86166,29.31166,null,51.919438,52.132633,null,14.058324,35.907757,null,52.132633,40.463667,null,20.593684,-25.274398,null,58.595272,61.92411,null,38.963745,46.603354,null,52.132633,37.09024,null,38.963745,39.074208,null,37.09024,36.204824,null,20.593684,-0.023559,null,45.943161,51.165691,null,20.593684,7.873054,null,35.86166,-38.416097,null,20.593684,36.204824,null,14.058324,35.86166,null,51.165691,37.09024,null,15.870032,49.817492,null,45.943161,41.87194,null,35.86166,23.685,null,23.424076,23.885942,null,52.132633,41.87194,null,48.379433,51.165691,null,63.397768,51.919438,null,63.397768,60.472024,null,40.463667,41.87194,null,14.058324,55.378051,null,63.397768,61.92411,null,35.86166,46.818188,null,36.204824,14.058324,null,46.603354,40.463667,null,46.603354,41.87194,null,20.593684,51.165691,null,50.850346,46.603354,null,14.058324,41.87194,null,20.593684,46.603354,null,15.870032,36.204824,null,22.396428,36.204824,null,51.165691,56.26392,null,30.375321,37.09024,null,35.86166,32.427908,null,20.593684,15.870032,null,35.86166,47.516231,null,14.058324,46.603354,null,38.963745,23.885942,null,20.593684,55.378051,null,46.603354,46.818188,null,36.204824,35.86166,null,46.603354,51.165691,null,51.919438,61.92411,null,35.86166,45.943161,null,14.058324,40.463667,null,37.09024,35.86166,null,35.907757,36.204824,null,63.397768,56.26392,null,37.09024,55.378051,null,49.817492,46.603354,null,38.963745,31.046051,null,35.86166,47.162494,null,35.86166,9.748917,null,48.379433,48.669026,null,38.963745,48.669026,null,35.86166,39.399872,null,41.87194,46.603354,null,35.86166,-0.023559,null,35.86166,42.315407,null,20.593684,4.210484,null,37.09024,22.396428,null,35.86166,-6.369028,null,20.593684,-6.369028,null,35.86166,58.595272,null,20.593684,52.132633,null,38.963745,55.378051,null,35.86166,9.30769,null,47.162494,51.165691,null,37.09024,18.735693,null,46.603354,45.943161,null,35.86166,7.946527,null,53.709807,61.52401,null,14.058324,51.919438,null,51.919438,47.162494,null,35.86166,13.794185,null,20.593684,40.463667,null,52.132633,51.919438,null,15.870032,1.352083,null,49.817492,51.919438,null,46.603354,50.850346,null,36.204824,4.210484,null,20.593684,8.619543,null,14.058324,52.132633,null,50.850346,49.817492,null,51.919438,48.669026,null,35.86166,33.854721,null,35.86166,55.169438,null,35.86166,-32.522779,null,51.919438,56.879635,null,35.86166,8.537981,null,38.963745,42.733883,null,40.463667,51.165691,null,35.86166,26.3351,null,38.963745,33.223191,null,35.86166,40.339852,null,-0.789275,36.204824,null,35.86166,48.019573,null,61.52401,53.709807,null,31.046051,37.09024,null,46.603354,37.09024,null,56.26392,63.397768,null,46.603354,55.378051,null,51.165691,47.162494,null,51.919438,45.943161,null,51.919438,42.733883,null,20.593684,56.130366,null,51.919438,55.169438,null,56.879635,63.397768,null,30.375321,51.165691,null,51.919438,48.379433,null,38.963745,63.397768,null,35.86166,45.1,null,38.963745,61.52401,null,39.399872,40.463667,null,23.424076,29.31166,null,51.919438,61.52401,null,35.86166,12.565679,null,38.963745,51.919438,null,35.907757,14.058324,null,35.907757,-0.789275,null,35.86166,28.0,null,39.399872,55.378051,null,35.86166,48.669026,null,-30.559482,55.378051,null,1.352083,35.86166,null,35.86166,-1.831239,null,52.132633,55.169438,null,20.593684,50.850346,null,14.058324,63.397768,null,48.379433,45.943161,null,44.016521,45.943161,null,47.411631,45.943161,null,45.943161,46.603354,null,51.165691,48.669026,null,35.86166,12.862807,null,23.634501,56.130366,null,61.52401,48.019573,null,55.169438,63.397768,null,52.132633,49.817492,null,37.09024,1.352083,null,1.352083,36.204824,null,-23.442503,-14.235004,null,1.352083,-25.274398,null,38.963745,50.850346,null,22.396428,55.378051,null,-17.713371,-7.109535,null,46.603354,33.886917,null,51.165691,38.963745,null,21.916221,35.907757,null,40.463667,31.791702,null,45.943161,40.463667,null,52.132633,63.397768,null,51.165691,61.52401,null,35.86166,25.354826,null,37.09024,-30.559482,null,35.907757,35.86166,null,36.204824,50.850346,null,20.593684,23.885942,null,14.058324,61.52401,null,14.058324,-30.559482,null,35.86166,4.535277,null,35.907757,37.09024,null,52.132633,56.26392,null,51.165691,53.41291,null,35.86166,15.552727,null,49.817492,48.669026,null,51.165691,61.92411,null,52.132633,36.204824,null,41.87194,51.165691,null,51.919438,46.603354,null,41.87194,39.074208,null,23.424076,12.879721,null,23.634501,15.783471,null,55.378051,37.09024,null,35.86166,30.585164,null,35.86166,28.394857,null,35.86166,26.02751,null,51.919438,55.378051,null,51.165691,26.820553,null,55.378051,11.825138,null,38.963745,56.130366,null,41.87194,9.081999,null,38.963745,42.315407,null,35.86166,31.791702,null,20.593684,-30.559482,null,35.86166,-2.1646,null,15.870032,-0.789275,null,35.86166,18.735693,null,51.919438,63.397768,null,15.870032,52.132633,null,22.396428,51.165691,null,51.165691,35.86166,null,47.411631,48.669026,null,43.915886,51.165691,null,51.165691,45.943161,null,45.943161,35.126413,null,53.41291,36.204824,null,46.603354,52.132633,null,23.424076,26.02751,null,4.210484,61.52401,null,-0.789275,23.885942,null,-14.235004,-38.416097,null,55.378051,52.132633,null,37.09024,-35.675147,null,38.963745,12.862807,null,63.397768,51.165691,null,4.570868,23.634501,null,35.86166,21.512583,null,50.850346,51.165691,null,30.375321,36.204824,null,37.09024,15.870032,null,38.963745,56.26392,null,37.09024,9.748917,null,21.916221,35.86166,null,42.733883,52.132633,null,20.593684,41.87194,null,35.86166,9.145,null,35.86166,15.783471,null,35.86166,41.377491,null,14.058324,47.516231,null,20.593684,49.817492,null,50.850346,40.463667,null,1.352083,15.870032,null,38.963745,33.886917,null,4.210484,1.352083,null,23.424076,21.512583,null,35.86166,33.886917,null,41.87194,40.463667,null,52.132633,38.963745,null,38.963745,45.943161,null,21.512583,5.152149,null,14.058324,23.424076,null,38.963745,47.162494,null,55.378051,46.603354,null,38.963745,-35.675147,null,56.26392,60.472024,null,-0.789275,-25.274398,null,48.669026,46.603354,null,38.963745,53.41291,null,35.907757,1.352083,null,22.396428,-25.274398,null,1.352083,35.907757,null,20.593684,14.497401,null,37.09024,35.907757,null,35.86166,42.733883,null,-40.900557,35.907757,null,46.603354,31.791702,null,63.397768,55.378051,null,23.885942,29.31166,null,15.870032,20.593684,null,38.963745,-0.023559,null,55.378051,56.26392,null,37.09024,52.132633,null,-25.274398,-40.900557,null,46.603354,35.86166,null,51.165691,60.472024,null,37.09024,20.593684,null,35.907757,22.396428,null,38.963745,40.463667,null,22.396428,14.058324,null,48.669026,49.817492,null,37.09024,-14.235004,null,51.165691,33.886917,null,35.86166,44.016521,null,55.378051,51.919438,null,14.058324,56.130366,null,51.165691,39.399872,null,48.669026,47.516231,null,35.86166,22.198745,null,15.870032,50.850346,null,37.09024,23.885942,null,35.907757,23.885942,null,23.424076,38.969719,null,14.058324,-25.274398,null,55.378051,40.463667,null,50.850346,55.378051,null,46.818188,51.165691,null,-30.559482,-22.95764,null,35.86166,15.199999,null,-9.189967,18.735693,null,20.593684,-0.789275,null,52.132633,23.634501,null,50.850346,47.516231,null,37.09024,46.818188,null,38.963745,23.634501,null,49.817492,61.52401,null,35.86166,34.802075,null,63.397768,50.850346,null,22.396428,52.132633,null,35.907757,18.971187,null,1.352083,-0.789275,null,51.919438,45.1,null,-11.202692,52.132633,null,51.165691,46.151241,null,46.603354,51.919438,null,51.165691,49.815273,null,35.86166,7.539989,null,20.593684,31.791702,null,22.396428,46.603354,null,35.86166,-11.202692,null,38.963745,48.379433,null,20.593684,-40.900557,null,42.733883,51.165691,null,55.378051,41.87194,null,4.210484,38.963745,null,46.603354,12.238333,null,51.165691,30.585164,null,37.09024,9.081999,null,55.169438,61.52401,null,41.87194,46.818188,null,38.963745,41.87194,null,35.86166,7.873054,null,50.850346,41.87194,null,52.132633,53.41291,null,35.86166,41.0,null,20.593684,35.907757,null,-2.1646,1.373333,null,45.943161,49.815273,null,35.907757,23.634501,null,35.86166,14.497401,null,20.593684,21.512583,null,36.204824,22.396428,null,1.352083,22.396428,null,22.396428,41.87194,null,52.132633,61.92411,null,-35.675147,-14.235004,null,51.165691,55.169438,null,46.603354,39.399872,null,47.411631,61.52401,null,51.165691,42.733883,null,44.016521,41.87194,null,51.165691,28.0,null,52.132633,23.424076,null,38.963745,40.143105,null,50.850346,53.41291,null,31.046051,55.378051,null,4.210484,46.603354,null,35.86166,11.825138,null,49.817492,50.850346,null,38.963745,28.0,null,47.162494,46.603354,null,35.907757,51.919438,null,23.424076,20.593684,null,51.165691,39.074208,null,15.870032,35.907757,null,20.593684,33.223191,null,22.396428,47.162494,null,37.09024,23.424076,null,55.169438,51.919438,null,12.879721,15.870032,null,35.86166,-6.314993,null,1.352083,4.210484,null,51.165691,23.424076,null,51.919438,56.26392,null,55.378051,50.850346,null,63.397768,38.963745,null,-23.442503,-38.416097,null,-1.940278,-2.1646,null,51.165691,44.016521,null,37.09024,25.354826,null,41.87194,37.09024,null,53.41291,55.378051,null,38.963745,48.019573,null,52.132633,46.818188,null,15.870032,63.397768,null,1.352083,56.130366,null,15.870032,-25.274398,null,38.963745,49.817492,null,55.378051,61.52401,null,15.870032,21.00789,null,35.907757,-25.274398,null,52.132633,47.516231,null,39.399872,56.26392,null,46.603354,1.352083,null,50.850346,46.818188,null,51.919438,40.463667,null,37.09024,61.52401,null,-25.274398,37.09024,null,38.963745,43.915886,null,20.593684,-18.665695,null,55.378051,26.820553,null,4.210484,23.634501,null,41.608635,52.132633,null,36.204824,37.09024,null,-30.559482,-19.015438,null,20.593684,63.397768,null,46.603354,63.397768,null,39.399872,46.603354,null,49.817492,52.132633,null,53.41291,51.165691,null,-38.416097,-32.522779,null,37.09024,-9.189967,null,4.210484,36.204824,null,46.151241,46.818188,null,40.463667,52.132633,null,49.817492,41.87194,null,38.963745,31.791702,null,20.593684,21.916221,null,38.963745,32.427908,null,48.669026,47.162494,null,55.169438,56.26392,null,45.1,51.165691,null,37.09024,-38.416097,null,20.593684,-18.766947,null,52.132633,60.472024,null,35.86166,7.369722,null,20.593684,28.394857,null,49.817492,37.09024,null,-0.789275,37.09024,null,14.058324,22.396428,null,55.169438,58.595272,null,37.09024,4.570868,null,38.963745,29.31166,null,51.919438,37.09024,null,37.09024,39.399872,null,41.87194,41.0,null,35.86166,41.20438,null,52.132633,40.143105,null,38.963745,44.016521,null,20.593684,51.919438,null,40.463667,16.5388,null,4.210484,51.165691,null,55.378051,1.352083,null,41.87194,31.046051,null,63.397768,41.87194,null,-25.274398,15.870032,null,21.512583,25.354826,null,51.165691,56.879635,null,20.593684,56.26392,null,20.593684,30.375321,null,22.396428,35.907757,null,50.850346,51.919438,null,35.86166,19.85627,null,20.593684,-26.522503,null,-40.900557,55.378051,null,38.963745,35.937496,null,20.593684,12.862807,null,49.817492,55.378051,null,58.595272,63.397768,null,51.919438,44.016521,null,56.26392,52.132633,null,45.943161,52.132633,null,56.26392,37.09024,null,14.058324,-0.789275,null,20.593684,26.02751,null,37.09024,25.03428,null,14.058324,23.634501,null,46.603354,47.516231,null,48.379433,51.919438,null,20.593684,22.396428,null,37.09024,41.87194,null,-9.189967,-1.831239,null,41.87194,55.378051,null,45.943161,48.669026,null,51.919438,41.87194,null,37.09024,12.879721,null,51.165691,45.1,null,56.26392,51.165691,null,46.603354,36.204824,null,35.86166,-20.348404,null,39.399872,-11.202692,null,20.593684,7.539989,null,20.593684,9.945587,null,52.132633,61.52401,null,14.058324,12.565679,null,38.963745,4.210484,null,36.204824,46.603354,null,49.817492,47.516231,null,41.87194,35.126413,null,31.791702,28.0,null,61.52401,51.919438,null,38.963745,35.126413,null,35.86166,-23.442503,null,35.907757,15.870032,null,4.210484,41.87194,null,35.907757,61.52401,null,40.463667,30.585164,null,20.593684,5.152149,null,35.86166,56.879635,null,47.162494,41.87194,null,35.86166,3.202778,null,22.396428,56.130366,null,13.794185,37.09024,null,38.963745,55.169438,null,46.603354,48.669026,null,61.52401,48.379433,null,35.907757,-32.522779,null,40.463667,15.783471,null,46.603354,22.396428,null,63.397768,37.09024,null,4.210484,12.565679,null,36.204824,-0.789275,null,22.396428,23.634501,null,58.595272,64.963051,null,46.603354,47.162494,null,51.919438,53.709807,null,4.210484,4.535277,null,52.132633,39.399872,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":0.5012786963955556},"lon":[104.195397,138.252924,null,104.195397,10.451526,null,104.195397,-3.435973,null,104.195397,5.291266,null,-102.552784,-95.712891,null,104.195397,-106.346771,null,-95.712891,-106.346771,null,104.195397,1.888334,null,104.195397,133.775136,null,104.195397,127.766922,null,108.277199,-95.712891,null,104.195397,-3.74922,null,5.291266,10.451526,null,104.195397,12.56738,null,104.195397,19.145136,null,104.195397,4.351721,null,104.195397,105.318756,null,104.195397,101.975766,null,78.96288,-95.712891,null,104.195397,78.96288,null,104.195397,45.079162,null,104.195397,-51.92528,null,104.195397,16.354896,null,10.451526,1.888334,null,104.195397,-102.552784,null,35.243322,-95.712891,null,104.195397,-71.542969,null,104.195397,114.109497,null,104.195397,43.679291,null,35.243322,5.291266,null,104.195397,108.277199,null,10.451526,19.145136,null,-70.162651,-95.712891,null,78.96288,53.847818,null,5.291266,4.351721,null,104.195397,53.847818,null,121.774017,138.252924,null,35.243322,10.451526,null,5.291266,1.888334,null,104.195397,121.774017,null,10.451526,14.550072,null,-7.09262,-3.74922,null,24.96676,19.145136,null,19.145136,10.451526,null,10.451526,5.291266,null,104.195397,100.992541,null,104.195397,113.921327,null,104.195397,103.819836,null,-95.712891,4.351721,null,104.195397,9.501785,null,10.451526,12.56738,null,104.195397,25.748151,null,-95.712891,133.775136,null,104.195397,174.885971,null,10.451526,15.472962,null,108.277199,10.451526,null,108.277199,138.252924,null,104.195397,14.995463,null,100.992541,-95.712891,null,104.195397,22.937506,null,100.992541,10.451526,null,10.451526,8.227512,null,104.195397,8.675277,null,114.109497,104.195397,null,-95.712891,10.451526,null,10.451526,-3.435973,null,35.243322,17.228331,null,104.195397,-75.015152,null,104.195397,-8.24389,null,104.195397,-74.297333,null,104.195397,21.824312,null,10.451526,-3.74922,null,1.888334,3.0,null,4.351721,5.291266,null,104.195397,95.955974,null,104.195397,34.851612,null,104.195397,35.243322,null,15.472962,10.451526,null,104.195397,8.468946,null,104.195397,30.802498,null,35.243322,28.369885,null,78.96288,65.0,null,-3.435973,10.451526,null,78.96288,8.675277,null,-3.435973,-8.24389,null,114.109497,-95.712891,null,-3.74922,1.888334,null,104.195397,31.16558,null,104.195397,15.827659,null,-3.74922,-8.224454,null,19.145136,15.472962,null,10.451526,16.354896,null,104.195397,15.472962,null,-51.92528,-58.443832,null,78.96288,90.3563,null,10.451526,4.351721,null,104.195397,69.345116,null,5.291266,-3.435973,null,104.195397,47.481766,null,19.145136,5.291266,null,108.277199,127.766922,null,5.291266,-3.74922,null,78.96288,133.775136,null,25.013607,25.748151,null,35.243322,1.888334,null,5.291266,-95.712891,null,35.243322,21.824312,null,-95.712891,138.252924,null,78.96288,37.906193,null,24.96676,10.451526,null,78.96288,80.771797,null,104.195397,-63.616672,null,78.96288,138.252924,null,108.277199,104.195397,null,10.451526,-95.712891,null,100.992541,15.472962,null,24.96676,12.56738,null,104.195397,90.3563,null,53.847818,45.079162,null,5.291266,12.56738,null,31.16558,10.451526,null,16.354896,19.145136,null,16.354896,8.468946,null,-3.74922,12.56738,null,108.277199,-3.435973,null,16.354896,25.748151,null,104.195397,8.227512,null,138.252924,108.277199,null,1.888334,-3.74922,null,1.888334,12.56738,null,78.96288,10.451526,null,4.351721,1.888334,null,108.277199,12.56738,null,78.96288,1.888334,null,100.992541,138.252924,null,114.109497,138.252924,null,10.451526,9.501785,null,69.345116,-95.712891,null,104.195397,53.688046,null,78.96288,100.992541,null,104.195397,14.550072,null,108.277199,1.888334,null,35.243322,45.079162,null,78.96288,-3.435973,null,1.888334,8.227512,null,138.252924,104.195397,null,1.888334,10.451526,null,19.145136,25.748151,null,104.195397,24.96676,null,108.277199,-3.74922,null,-95.712891,104.195397,null,127.766922,138.252924,null,16.354896,9.501785,null,-95.712891,-3.435973,null,15.472962,1.888334,null,35.243322,34.851612,null,104.195397,19.503304,null,104.195397,-83.753428,null,31.16558,19.699024,null,35.243322,19.699024,null,104.195397,-8.224454,null,12.56738,1.888334,null,104.195397,37.906193,null,104.195397,43.356892,null,78.96288,101.975766,null,-95.712891,114.109497,null,104.195397,34.888822,null,78.96288,34.888822,null,104.195397,25.013607,null,78.96288,5.291266,null,35.243322,-3.435973,null,104.195397,2.315834,null,19.503304,10.451526,null,-95.712891,-70.162651,null,1.888334,24.96676,null,104.195397,-1.023194,null,27.953389,105.318756,null,108.277199,19.145136,null,19.145136,19.503304,null,104.195397,-88.89653,null,78.96288,-3.74922,null,5.291266,19.145136,null,100.992541,103.819836,null,15.472962,19.145136,null,1.888334,4.351721,null,138.252924,101.975766,null,78.96288,0.824782,null,108.277199,5.291266,null,4.351721,15.472962,null,19.145136,19.699024,null,104.195397,35.862285,null,104.195397,23.881275,null,104.195397,-55.765835,null,19.145136,24.603189,null,104.195397,-80.782127,null,35.243322,25.48583,null,-3.74922,10.451526,null,104.195397,17.228331,null,35.243322,43.679291,null,104.195397,127.510093,null,113.921327,138.252924,null,104.195397,66.923684,null,105.318756,27.953389,null,34.851612,-95.712891,null,1.888334,-95.712891,null,9.501785,16.354896,null,1.888334,-3.435973,null,10.451526,19.503304,null,19.145136,24.96676,null,19.145136,25.48583,null,78.96288,-106.346771,null,19.145136,23.881275,null,24.603189,16.354896,null,69.345116,10.451526,null,19.145136,31.16558,null,35.243322,16.354896,null,104.195397,15.2,null,35.243322,105.318756,null,-8.224454,-3.74922,null,53.847818,47.481766,null,19.145136,105.318756,null,104.195397,104.990963,null,35.243322,19.145136,null,127.766922,108.277199,null,127.766922,113.921327,null,104.195397,3.0,null,-8.224454,-3.435973,null,104.195397,19.699024,null,22.937506,-3.435973,null,103.819836,104.195397,null,104.195397,-78.183406,null,5.291266,23.881275,null,78.96288,4.351721,null,108.277199,16.354896,null,31.16558,24.96676,null,21.005859,24.96676,null,28.369885,24.96676,null,24.96676,1.888334,null,10.451526,19.699024,null,104.195397,30.217636,null,-102.552784,-106.346771,null,105.318756,66.923684,null,23.881275,16.354896,null,5.291266,15.472962,null,-95.712891,103.819836,null,103.819836,138.252924,null,-58.443832,-51.92528,null,103.819836,133.775136,null,35.243322,4.351721,null,114.109497,-3.435973,null,178.065033,179.194167,null,1.888334,9.537499,null,10.451526,35.243322,null,95.955974,127.766922,null,-3.74922,-7.09262,null,24.96676,-3.74922,null,5.291266,16.354896,null,10.451526,105.318756,null,104.195397,51.183884,null,-95.712891,22.937506,null,127.766922,104.195397,null,138.252924,4.351721,null,78.96288,45.079162,null,108.277199,105.318756,null,108.277199,22.937506,null,104.195397,114.727669,null,127.766922,-95.712891,null,5.291266,9.501785,null,10.451526,-8.24389,null,104.195397,48.516388,null,15.472962,19.699024,null,10.451526,25.748151,null,5.291266,138.252924,null,12.56738,10.451526,null,19.145136,1.888334,null,12.56738,21.824312,null,53.847818,121.774017,null,-102.552784,-90.230759,null,-3.435973,-95.712891,null,104.195397,36.238414,null,104.195397,84.124008,null,104.195397,50.55096,null,19.145136,-3.435973,null,10.451526,30.802498,null,-3.435973,42.590275,null,35.243322,-106.346771,null,12.56738,8.675277,null,35.243322,43.356892,null,104.195397,-7.09262,null,78.96288,22.937506,null,104.195397,24.15536,null,100.992541,113.921327,null,104.195397,-70.162651,null,19.145136,16.354896,null,100.992541,5.291266,null,114.109497,10.451526,null,10.451526,104.195397,null,28.369885,19.699024,null,17.679076,10.451526,null,10.451526,24.96676,null,24.96676,33.429859,null,-8.24389,138.252924,null,1.888334,5.291266,null,53.847818,50.55096,null,101.975766,105.318756,null,113.921327,45.079162,null,-51.92528,-63.616672,null,-3.435973,5.291266,null,-95.712891,-71.542969,null,35.243322,30.217636,null,16.354896,10.451526,null,-74.297333,-102.552784,null,104.195397,55.923255,null,4.351721,10.451526,null,69.345116,138.252924,null,-95.712891,100.992541,null,35.243322,9.501785,null,-95.712891,-83.753428,null,95.955974,104.195397,null,25.48583,5.291266,null,78.96288,12.56738,null,104.195397,40.489673,null,104.195397,-90.230759,null,104.195397,64.585262,null,108.277199,14.550072,null,78.96288,15.472962,null,4.351721,-3.74922,null,103.819836,100.992541,null,35.243322,9.537499,null,101.975766,103.819836,null,53.847818,55.923255,null,104.195397,9.537499,null,12.56738,-3.74922,null,5.291266,35.243322,null,35.243322,24.96676,null,55.923255,46.199616,null,108.277199,53.847818,null,35.243322,19.503304,null,-3.435973,1.888334,null,35.243322,-71.542969,null,9.501785,8.468946,null,113.921327,133.775136,null,19.699024,1.888334,null,35.243322,-8.24389,null,127.766922,103.819836,null,114.109497,133.775136,null,103.819836,127.766922,null,78.96288,-14.452362,null,-95.712891,127.766922,null,104.195397,25.48583,null,174.885971,127.766922,null,1.888334,-7.09262,null,16.354896,-3.435973,null,45.079162,47.481766,null,100.992541,78.96288,null,35.243322,37.906193,null,-3.435973,9.501785,null,-95.712891,5.291266,null,133.775136,174.885971,null,1.888334,104.195397,null,10.451526,8.468946,null,-95.712891,78.96288,null,127.766922,114.109497,null,35.243322,-3.74922,null,114.109497,108.277199,null,19.699024,15.472962,null,-95.712891,-51.92528,null,10.451526,9.537499,null,104.195397,21.005859,null,-3.435973,19.145136,null,108.277199,-106.346771,null,10.451526,-8.224454,null,19.699024,14.550072,null,104.195397,113.543873,null,100.992541,4.351721,null,-95.712891,45.079162,null,127.766922,45.079162,null,53.847818,59.556278,null,108.277199,133.775136,null,-3.435973,-3.74922,null,4.351721,-3.435973,null,8.227512,10.451526,null,22.937506,18.49041,null,104.195397,-86.241905,null,-75.015152,-70.162651,null,78.96288,113.921327,null,5.291266,-102.552784,null,4.351721,14.550072,null,-95.712891,8.227512,null,35.243322,-102.552784,null,15.472962,105.318756,null,104.195397,38.996815,null,16.354896,4.351721,null,114.109497,5.291266,null,127.766922,-72.285215,null,103.819836,113.921327,null,19.145136,15.2,null,17.873887,5.291266,null,10.451526,14.995463,null,1.888334,19.145136,null,10.451526,6.129583,null,104.195397,-5.54708,null,78.96288,-7.09262,null,114.109497,1.888334,null,104.195397,17.873887,null,35.243322,31.16558,null,78.96288,174.885971,null,25.48583,10.451526,null,-3.435973,12.56738,null,101.975766,35.243322,null,1.888334,-1.561593,null,10.451526,36.238414,null,-95.712891,8.675277,null,23.881275,105.318756,null,12.56738,8.227512,null,35.243322,12.56738,null,104.195397,80.771797,null,4.351721,12.56738,null,5.291266,-8.24389,null,104.195397,20.0,null,78.96288,127.766922,null,24.15536,32.290275,null,24.96676,6.129583,null,127.766922,-102.552784,null,104.195397,-14.452362,null,78.96288,55.923255,null,138.252924,114.109497,null,103.819836,114.109497,null,114.109497,12.56738,null,5.291266,25.748151,null,-71.542969,-51.92528,null,10.451526,23.881275,null,1.888334,-8.224454,null,28.369885,105.318756,null,10.451526,25.48583,null,21.005859,12.56738,null,10.451526,3.0,null,5.291266,53.847818,null,35.243322,47.576927,null,4.351721,-8.24389,null,34.851612,-3.435973,null,101.975766,1.888334,null,104.195397,42.590275,null,15.472962,4.351721,null,35.243322,3.0,null,19.503304,1.888334,null,127.766922,19.145136,null,53.847818,78.96288,null,10.451526,21.824312,null,100.992541,127.766922,null,78.96288,43.679291,null,114.109497,19.503304,null,-95.712891,53.847818,null,23.881275,19.145136,null,121.774017,100.992541,null,104.195397,143.95555,null,103.819836,101.975766,null,10.451526,53.847818,null,19.145136,9.501785,null,-3.435973,4.351721,null,16.354896,35.243322,null,-58.443832,-63.616672,null,29.873888,24.15536,null,10.451526,21.005859,null,-95.712891,51.183884,null,12.56738,-95.712891,null,-8.24389,-3.435973,null,35.243322,66.923684,null,5.291266,8.227512,null,100.992541,16.354896,null,103.819836,-106.346771,null,100.992541,133.775136,null,35.243322,15.472962,null,-3.435973,105.318756,null,100.992541,-10.940835,null,127.766922,133.775136,null,5.291266,14.550072,null,-8.224454,9.501785,null,1.888334,103.819836,null,4.351721,8.227512,null,19.145136,-3.74922,null,-95.712891,105.318756,null,133.775136,-95.712891,null,35.243322,17.679076,null,78.96288,35.529562,null,-3.435973,30.802498,null,101.975766,-102.552784,null,21.745275,5.291266,null,138.252924,-95.712891,null,22.937506,29.154857,null,78.96288,16.354896,null,1.888334,16.354896,null,-8.224454,1.888334,null,15.472962,5.291266,null,-8.24389,10.451526,null,-63.616672,-55.765835,null,-95.712891,-75.015152,null,101.975766,138.252924,null,14.995463,8.227512,null,-3.74922,5.291266,null,15.472962,12.56738,null,35.243322,-7.09262,null,78.96288,95.955974,null,35.243322,53.688046,null,19.699024,19.503304,null,23.881275,9.501785,null,15.2,10.451526,null,-95.712891,-63.616672,null,78.96288,46.869107,null,5.291266,8.468946,null,104.195397,12.354722,null,78.96288,84.124008,null,15.472962,-95.712891,null,113.921327,-95.712891,null,108.277199,114.109497,null,23.881275,25.013607,null,-95.712891,-74.297333,null,35.243322,47.481766,null,19.145136,-95.712891,null,-95.712891,-8.224454,null,12.56738,20.0,null,104.195397,74.766098,null,5.291266,47.576927,null,35.243322,21.005859,null,78.96288,19.145136,null,-3.74922,-23.0418,null,101.975766,10.451526,null,-3.435973,103.819836,null,12.56738,34.851612,null,16.354896,12.56738,null,133.775136,100.992541,null,55.923255,51.183884,null,10.451526,24.603189,null,78.96288,9.501785,null,78.96288,69.345116,null,114.109497,127.766922,null,4.351721,19.145136,null,104.195397,102.495496,null,78.96288,31.465866,null,174.885971,-3.435973,null,35.243322,14.375416,null,78.96288,30.217636,null,15.472962,-3.435973,null,25.013607,16.354896,null,19.145136,21.005859,null,9.501785,5.291266,null,24.96676,5.291266,null,9.501785,-95.712891,null,108.277199,113.921327,null,78.96288,50.55096,null,-95.712891,-77.39628,null,108.277199,-102.552784,null,1.888334,14.550072,null,31.16558,19.145136,null,78.96288,114.109497,null,-95.712891,12.56738,null,-75.015152,-78.183406,null,12.56738,-3.435973,null,24.96676,19.699024,null,19.145136,12.56738,null,-95.712891,121.774017,null,10.451526,15.2,null,9.501785,10.451526,null,1.888334,138.252924,null,104.195397,57.552152,null,-8.224454,17.873887,null,78.96288,-5.54708,null,78.96288,-9.696645,null,5.291266,105.318756,null,108.277199,104.990963,null,35.243322,101.975766,null,138.252924,1.888334,null,15.472962,14.550072,null,12.56738,33.429859,null,-7.09262,3.0,null,105.318756,19.145136,null,35.243322,33.429859,null,104.195397,-58.443832,null,127.766922,100.992541,null,101.975766,12.56738,null,127.766922,105.318756,null,-3.74922,36.238414,null,78.96288,46.199616,null,104.195397,24.603189,null,19.503304,12.56738,null,104.195397,73.22068,null,114.109497,-106.346771,null,-88.89653,-95.712891,null,35.243322,23.881275,null,1.888334,19.699024,null,105.318756,31.16558,null,127.766922,-55.765835,null,-3.74922,-90.230759,null,1.888334,114.109497,null,16.354896,-95.712891,null,101.975766,104.990963,null,138.252924,113.921327,null,114.109497,-102.552784,null,25.013607,-19.020835,null,1.888334,19.503304,null,19.145136,27.953389,null,101.975766,114.727669,null,5.291266,-8.224454,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,46.603354,-11.202692,-30.559482,55.378051,40.143105,56.130366,42.315407,39.399872,-38.416097,-16.290154,-14.235004,-35.675147,4.570868,9.748917,15.783471,15.199999,41.87194,36.204824,-23.442503,-9.189967,40.463667,-32.522779,-25.274398,4.535277,35.86166,56.26392,-17.713371,61.92411,51.165691,31.046051,7.539989,19.85627,4.210484,52.132633,-20.904305,-40.900557,60.472024,-6.314993,12.879721,25.354826,20.593684,1.352083,15.870032,38.963745,-6.369028,37.09024,26.02751,29.31166,23.885942,23.424076,40.069099,61.52401,13.193887,17.060816,25.03428,19.3133,18.218785,13.909444,50.850346,33.0,28.0,47.516231,43.915886,42.733883,7.369722,-0.228021,-2.1646,45.1,35.126413,49.817492,58.595272,-0.803689,13.443182,7.946527,39.074208,9.945587,22.396428,47.162494,64.963051,33.223191,53.41291,18.109581,48.019573,33.854721,56.879635,6.428055,55.169438,49.815273,-18.766947,17.570692,35.937496,21.00789,23.634501,47.411631,31.791702,21.512583,30.375321,8.537981,51.919438,45.943161,-1.940278,14.497401,44.016521,-4.679574,48.669026,46.151241,63.397768,46.818188,33.886917,48.379433,41.608635,26.820553,12.238333,-13.133897,-1.831239,4.860416,12.984305,17.189877,-0.789275,21.916221,35.907757,53.709807,38.861034,38.969719,12.565679,18.04248,3.919305,18.735693,13.794185,41.0,23.685,-22.328474,-9.64571,-3.373056,16.5388,7.873054,15.454166,21.521757,9.30769,15.414999,9.145,-17.679742,11.825138,31.952162,71.706936,18.971187,32.427908,30.585164,-0.023559,40.339852,41.20438,26.3351,22.198745,-13.254308,3.202778,-20.348404,46.862496,42.708678,-18.665695,-22.95764,28.394857,12.16957,12.52111,-15.376706,12.865416,9.081999,-8.874217,8.460555,14.058324,5.152149,-19.015438,12.862807,34.802075,8.619543,-21.178986,10.691803,1.373333,41.377491,6.42375,15.552727,7.131474,36.140751,61.892635,-0.522778,6.611111,-11.6455,13.444304,41.902916,17.607789,17.664332,46.946947,-13.768752,43.94236,-21.236736,7.862684,7.425554,-14.28522,-19.054445,-49.280366,1.650801,12.20189,11.803749,17.897476,0.18636,27.514162,-29.609988,-26.522503,18.420695,12.262776,32.3078,-51.796253,21.694025,-13.759029,-29.040835,15.179384,-15.965,10.961632,17.357822,-3.370417,-7.109535,7.51498,16.742498,-6.343194,-90.0,24.215527],"lon":[1.601554,1.888334,17.873887,22.937506,-3.435973,47.576927,-106.346771,43.356892,-8.224454,-63.616672,-63.588653,-51.92528,-71.542969,-74.297333,-83.753428,-90.230759,-86.241905,12.56738,138.252924,-58.443832,-75.015152,-3.74922,-55.765835,133.775136,114.727669,104.195397,9.501785,178.065033,25.748151,10.451526,34.851612,-5.54708,102.495496,101.975766,5.291266,165.618042,174.885971,8.468946,143.95555,121.774017,51.183884,78.96288,103.819836,100.992541,35.243322,34.888822,-95.712891,50.55096,47.481766,45.079162,53.847818,45.038189,105.318756,-59.543198,-61.796428,-77.39628,-81.2546,-63.043653,-60.978893,4.351721,65.0,3.0,14.550072,17.679076,25.48583,12.354722,15.827659,24.15536,15.2,33.429859,15.472962,25.013607,11.609444,-15.310139,-1.023194,21.824312,-9.696645,114.109497,19.503304,-19.020835,43.679291,-8.24389,-77.297508,66.923684,35.862285,24.603189,-9.429499,23.881275,6.129583,46.869107,-3.996166,14.375416,-10.940835,-102.552784,28.369885,-7.09262,55.923255,69.345116,-80.782127,19.145136,24.96676,29.873888,-14.452362,21.005859,55.491977,19.699024,14.995463,16.354896,8.227512,9.537499,31.16558,21.745275,30.802498,-1.561593,27.849332,-78.183406,-58.93018,-61.287228,-88.49765,113.921327,95.955974,127.766922,27.953389,71.276093,59.556278,104.990963,-63.05483,-56.027783,-70.162651,-88.89653,20.0,90.3563,24.684866,160.156194,29.918886,-23.0418,80.771797,18.732207,-77.781167,2.315834,-61.370976,40.489673,-149.406843,42.590275,35.233154,-42.604303,-72.285215,53.688046,36.238414,37.906193,127.510093,74.766098,17.228331,113.543873,34.301525,73.22068,57.552152,103.846656,19.37439,35.529562,18.49041,84.124008,-68.990021,-69.968338,166.959158,-85.207229,8.675277,125.727,-11.779889,108.277199,46.199616,29.154857,30.217636,38.996815,0.824782,-175.198242,-61.222503,32.290275,64.585262,-66.58973,48.516388,171.184478,-5.353585,-6.911806,166.931503,20.939444,43.3333,144.793731,12.453389,8.081666,145.94351,-56.32509,-177.156097,12.457777,-159.777671,30.217636,150.550812,-170.70444,-169.867233,69.348557,10.267895,-68.262383,-15.180413,-62.83055,6.613081,90.433601,28.233608,31.465866,-64.639968,-61.604171,-64.7505,-59.523613,-71.797928,-172.104629,167.954712,39.782334,-5.7089,-169.09022,-62.782998,-168.734039,179.194167,134.58252,-62.187366,71.876519,0.0,-12.885834],"marker":{"color":"red","opacity":0.8,"size":[6.8,20.0,18.8,20.0,20.0,17.6,20.0,20.0,20.0,20.0,12.8,20.0,20.0,20.0,20.0,16.4,11.6,20.0,20.0,12.0,20.0,20.0,12.8,20.0,12.0,20.0,20.0,15.2,20.0,20.0,20.0,18.4,7.6,20.0,20.0,7.2,20.0,20.0,7.6,20.0,15.2,20.0,20.0,20.0,20.0,18.4,20.0,18.0,17.6,20.0,20.0,13.6,20.0,16.0,7.2,8.4,4.8,2.4,5.2,20.0,8.0,10.0,20.0,20.0,20.0,10.0,11.6,13.6,20.0,17.6,20.0,20.0,7.6,5.2,15.2,20.0,7.6,20.0,20.0,15.2,13.6,20.0,9.6,20.0,16.4,20.0,8.8,20.0,20.0,13.2,11.6,16.0,6.4,20.0,20.0,20.0,20.0,20.0,19.6,20.0,20.0,12.0,13.6,20.0,8.4,20.0,20.0,20.0,20.0,10.4,20.0,20.0,18.4,8.4,10.0,18.8,7.6,4.4,6.4,20.0,13.2,20.0,20.0,6.4,6.8,11.2,4.4,6.8,20.0,14.8,12.4,11.6,5.6,4.0,2.8,3.2,13.2,5.2,9.6,6.4,2.8,10.4,7.6,6.4,2.0,5.6,5.6,12.8,14.8,20.0,1.2,15.6,11.2,9.2,4.0,11.6,17.2,16.0,13.2,10.4,8.0,6.0,7.2,6.4,3.6,8.0,15.2,2.0,5.2,20.0,6.4,8.0,6.4,4.4,8.4,2.4,20.0,11.6,10.4,7.2,4.4,3.2,5.6,4.0,0.8,3.2,3.6,4.0,0.4,4.4,2.4,0.8,1.2,4.0,3.2,4.8,3.2,0.4,0.4,0.8,5.6,1.6,2.0,2.0,2.8,2.4,2.8,8.0,4.8,5.2,6.0,0.4,3.2,2.4,1.2,2.0,0.4,0.4,4.0,1.6,0.8,1.6,2.0,0.4,0.8,0.4]},"mode":"markers","name":"2018","text":["AND\u003cbr\u003eConex\u00f5es: 17","FRA\u003cbr\u003eConex\u00f5es: 246","AGO\u003cbr\u003eConex\u00f5es: 47","ZAF\u003cbr\u003eConex\u00f5es: 163","GBR\u003cbr\u003eConex\u00f5es: 217","AZE\u003cbr\u003eConex\u00f5es: 44","CAN\u003cbr\u003eConex\u00f5es: 136","GEO\u003cbr\u003eConex\u00f5es: 57","PRT\u003cbr\u003eConex\u00f5es: 176","ARG\u003cbr\u003eConex\u00f5es: 51","BOL\u003cbr\u003eConex\u00f5es: 32","BRA\u003cbr\u003eConex\u00f5es: 133","CHL\u003cbr\u003eConex\u00f5es: 70","COL\u003cbr\u003eConex\u00f5es: 90","CRI\u003cbr\u003eConex\u00f5es: 52","GTM\u003cbr\u003eConex\u00f5es: 41","HND\u003cbr\u003eConex\u00f5es: 29","ITA\u003cbr\u003eConex\u00f5es: 198","JPN\u003cbr\u003eConex\u00f5es: 153","PRY\u003cbr\u003eConex\u00f5es: 30","PER\u003cbr\u003eConex\u00f5es: 83","ESP\u003cbr\u003eConex\u00f5es: 199","URY\u003cbr\u003eConex\u00f5es: 32","AUS\u003cbr\u003eConex\u00f5es: 142","BRN\u003cbr\u003eConex\u00f5es: 30","CHN\u003cbr\u003eConex\u00f5es: 272","DNK\u003cbr\u003eConex\u00f5es: 160","FJI\u003cbr\u003eConex\u00f5es: 38","FIN\u003cbr\u003eConex\u00f5es: 135","DEU\u003cbr\u003eConex\u00f5es: 254","ISR\u003cbr\u003eConex\u00f5es: 83","CIV\u003cbr\u003eConex\u00f5es: 46","LAO\u003cbr\u003eConex\u00f5es: 19","MYS\u003cbr\u003eConex\u00f5es: 93","NLD\u003cbr\u003eConex\u00f5es: 228","NCL\u003cbr\u003eConex\u00f5es: 18","NZL\u003cbr\u003eConex\u00f5es: 94","NOR\u003cbr\u003eConex\u00f5es: 119","PNG\u003cbr\u003eConex\u00f5es: 19","PHL\u003cbr\u003eConex\u00f5es: 72","QAT\u003cbr\u003eConex\u00f5es: 38","IND\u003cbr\u003eConex\u00f5es: 218","SGP\u003cbr\u003eConex\u00f5es: 151","THA\u003cbr\u003eConex\u00f5es: 166","TUR\u003cbr\u003eConex\u00f5es: 204","TZA\u003cbr\u003eConex\u00f5es: 46","USA\u003cbr\u003eConex\u00f5es: 239","BHR\u003cbr\u003eConex\u00f5es: 45","KWT\u003cbr\u003eConex\u00f5es: 44","SAU\u003cbr\u003eConex\u00f5es: 63","ARE\u003cbr\u003eConex\u00f5es: 188","ARM\u003cbr\u003eConex\u00f5es: 34","RUS\u003cbr\u003eConex\u00f5es: 152","BRB\u003cbr\u003eConex\u00f5es: 40","ATG\u003cbr\u003eConex\u00f5es: 18","BHS\u003cbr\u003eConex\u00f5es: 21","CYM\u003cbr\u003eConex\u00f5es: 12","AIA\u003cbr\u003eConex\u00f5es: 6","LCA\u003cbr\u003eConex\u00f5es: 13","BEL\u003cbr\u003eConex\u00f5es: 207","AFG\u003cbr\u003eConex\u00f5es: 20","DZA\u003cbr\u003eConex\u00f5es: 25","AUT\u003cbr\u003eConex\u00f5es: 55","BIH\u003cbr\u003eConex\u00f5es: 59","BGR\u003cbr\u003eConex\u00f5es: 93","CMR\u003cbr\u003eConex\u00f5es: 25","COG\u003cbr\u003eConex\u00f5es: 29","COD\u003cbr\u003eConex\u00f5es: 34","HRV\u003cbr\u003eConex\u00f5es: 102","CYP\u003cbr\u003eConex\u00f5es: 44","CZE\u003cbr\u003eConex\u00f5es: 165","EST\u003cbr\u003eConex\u00f5es: 89","GAB\u003cbr\u003eConex\u00f5es: 19","GMB\u003cbr\u003eConex\u00f5es: 13","GHA\u003cbr\u003eConex\u00f5es: 38","GRC\u003cbr\u003eConex\u00f5es: 127","GIN\u003cbr\u003eConex\u00f5es: 19","HKG\u003cbr\u003eConex\u00f5es: 173","HUN\u003cbr\u003eConex\u00f5es: 132","ISL\u003cbr\u003eConex\u00f5es: 38","IRQ\u003cbr\u003eConex\u00f5es: 34","IRL\u003cbr\u003eConex\u00f5es: 104","JAM\u003cbr\u003eConex\u00f5es: 24","KAZ\u003cbr\u003eConex\u00f5es: 65","LBN\u003cbr\u003eConex\u00f5es: 41","LVA\u003cbr\u003eConex\u00f5es: 95","LBR\u003cbr\u003eConex\u00f5es: 22","LTU\u003cbr\u003eConex\u00f5es: 114","LUX\u003cbr\u003eConex\u00f5es: 77","MDG\u003cbr\u003eConex\u00f5es: 33","MLI\u003cbr\u003eConex\u00f5es: 29","MLT\u003cbr\u003eConex\u00f5es: 40","MRT\u003cbr\u003eConex\u00f5es: 16","MEX\u003cbr\u003eConex\u00f5es: 70","MDA\u003cbr\u003eConex\u00f5es: 51","MAR\u003cbr\u003eConex\u00f5es: 112","OMN\u003cbr\u003eConex\u00f5es: 54","PAK\u003cbr\u003eConex\u00f5es: 148","PAN\u003cbr\u003eConex\u00f5es: 49","POL\u003cbr\u003eConex\u00f5es: 186","ROU\u003cbr\u003eConex\u00f5es: 116","RWA\u003cbr\u003eConex\u00f5es: 30","SEN\u003cbr\u003eConex\u00f5es: 34","SRB\u003cbr\u003eConex\u00f5es: 98","SYC\u003cbr\u003eConex\u00f5es: 21","SVK\u003cbr\u003eConex\u00f5es: 114","SVN\u003cbr\u003eConex\u00f5es: 121","SWE\u003cbr\u003eConex\u00f5es: 178","CHE\u003cbr\u003eConex\u00f5es: 217","TUN\u003cbr\u003eConex\u00f5es: 26","UKR\u003cbr\u003eConex\u00f5es: 116","MKD\u003cbr\u003eConex\u00f5es: 51","EGY\u003cbr\u003eConex\u00f5es: 46","BFA\u003cbr\u003eConex\u00f5es: 21","ZMB\u003cbr\u003eConex\u00f5es: 25","ECU\u003cbr\u003eConex\u00f5es: 47","GUY\u003cbr\u003eConex\u00f5es: 19","VCT\u003cbr\u003eConex\u00f5es: 11","BLZ\u003cbr\u003eConex\u00f5es: 16","IDN\u003cbr\u003eConex\u00f5es: 147","MMR\u003cbr\u003eConex\u00f5es: 33","KOR\u003cbr\u003eConex\u00f5es: 174","BLR\u003cbr\u003eConex\u00f5es: 71","TJK\u003cbr\u003eConex\u00f5es: 16","TKM\u003cbr\u003eConex\u00f5es: 17","KHM\u003cbr\u003eConex\u00f5es: 28","SXM\u003cbr\u003eConex\u00f5es: 11","SUR\u003cbr\u003eConex\u00f5es: 17","DOM\u003cbr\u003eConex\u00f5es: 61","SLV\u003cbr\u003eConex\u00f5es: 37","ALB\u003cbr\u003eConex\u00f5es: 31","BGD\u003cbr\u003eConex\u00f5es: 29","BWA\u003cbr\u003eConex\u00f5es: 14","SLB\u003cbr\u003eConex\u00f5es: 10","BDI\u003cbr\u003eConex\u00f5es: 7","CPV\u003cbr\u003eConex\u00f5es: 8","LKA\u003cbr\u003eConex\u00f5es: 33","TCD\u003cbr\u003eConex\u00f5es: 13","CUB\u003cbr\u003eConex\u00f5es: 24","BEN\u003cbr\u003eConex\u00f5es: 16","DMA\u003cbr\u003eConex\u00f5es: 7","ETH\u003cbr\u003eConex\u00f5es: 26","PYF\u003cbr\u003eConex\u00f5es: 19","DJI\u003cbr\u003eConex\u00f5es: 16","PSE\u003cbr\u003eConex\u00f5es: 5","GRL\u003cbr\u003eConex\u00f5es: 14","HTI\u003cbr\u003eConex\u00f5es: 14","IRN\u003cbr\u003eConex\u00f5es: 32","JOR\u003cbr\u003eConex\u00f5es: 37","KEN\u003cbr\u003eConex\u00f5es: 67","PRK\u003cbr\u003eConex\u00f5es: 3","KGZ\u003cbr\u003eConex\u00f5es: 39","LBY\u003cbr\u003eConex\u00f5es: 28","MAC\u003cbr\u003eConex\u00f5es: 23","MWI\u003cbr\u003eConex\u00f5es: 10","MDV\u003cbr\u003eConex\u00f5es: 29","MUS\u003cbr\u003eConex\u00f5es: 43","MNG\u003cbr\u003eConex\u00f5es: 40","MNE\u003cbr\u003eConex\u00f5es: 33","MOZ\u003cbr\u003eConex\u00f5es: 26","NAM\u003cbr\u003eConex\u00f5es: 20","NPL\u003cbr\u003eConex\u00f5es: 15","CUW\u003cbr\u003eConex\u00f5es: 18","ABW\u003cbr\u003eConex\u00f5es: 16","VUT\u003cbr\u003eConex\u00f5es: 9","NIC\u003cbr\u003eConex\u00f5es: 20","NGA\u003cbr\u003eConex\u00f5es: 38","TLS\u003cbr\u003eConex\u00f5es: 5","SLE\u003cbr\u003eConex\u00f5es: 13","VNM\u003cbr\u003eConex\u00f5es: 110","SOM\u003cbr\u003eConex\u00f5es: 16","ZWE\u003cbr\u003eConex\u00f5es: 20","SDN\u003cbr\u003eConex\u00f5es: 16","SYR\u003cbr\u003eConex\u00f5es: 11","TGO\u003cbr\u003eConex\u00f5es: 21","TON\u003cbr\u003eConex\u00f5es: 6","TTO\u003cbr\u003eConex\u00f5es: 55","UGA\u003cbr\u003eConex\u00f5es: 29","UZB\u003cbr\u003eConex\u00f5es: 26","VEN\u003cbr\u003eConex\u00f5es: 18","YEM\u003cbr\u003eConex\u00f5es: 11","MHL\u003cbr\u003eConex\u00f5es: 8","GIB\u003cbr\u003eConex\u00f5es: 14","FRO\u003cbr\u003eConex\u00f5es: 10","NRU\u003cbr\u003eConex\u00f5es: 2","CAF\u003cbr\u003eConex\u00f5es: 8","COM\u003cbr\u003eConex\u00f5es: 9","GUM\u003cbr\u003eConex\u00f5es: 10","VAT\u003cbr\u003eConex\u00f5es: 1","NER\u003cbr\u003eConex\u00f5es: 11","MNP\u003cbr\u003eConex\u00f5es: 6","SPM\u003cbr\u003eConex\u00f5es: 2","WLF\u003cbr\u003eConex\u00f5es: 3","SMR\u003cbr\u003eConex\u00f5es: 10","COK\u003cbr\u003eConex\u00f5es: 8","SSD\u003cbr\u003eConex\u00f5es: 12","FSM\u003cbr\u003eConex\u00f5es: 8","ASM\u003cbr\u003eConex\u00f5es: 1","NIU\u003cbr\u003eConex\u00f5es: 1","ATF\u003cbr\u003eConex\u00f5es: 2","GNQ\u003cbr\u003eConex\u00f5es: 14","BES\u003cbr\u003eConex\u00f5es: 4","GNB\u003cbr\u003eConex\u00f5es: 5","BLM\u003cbr\u003eConex\u00f5es: 5","STP\u003cbr\u003eConex\u00f5es: 7","BTN\u003cbr\u003eConex\u00f5es: 6","LSO\u003cbr\u003eConex\u00f5es: 7","SWZ\u003cbr\u003eConex\u00f5es: 20","VGB\u003cbr\u003eConex\u00f5es: 12","GRD\u003cbr\u003eConex\u00f5es: 13","BMU\u003cbr\u003eConex\u00f5es: 15","FLK\u003cbr\u003eConex\u00f5es: 1","TCA\u003cbr\u003eConex\u00f5es: 8","WSM\u003cbr\u003eConex\u00f5es: 6","NFK\u003cbr\u003eConex\u00f5es: 3","ERI\u003cbr\u003eConex\u00f5es: 5","SHN\u003cbr\u003eConex\u00f5es: 1","UMI\u003cbr\u003eConex\u00f5es: 1","KNA\u003cbr\u003eConex\u00f5es: 10","KIR\u003cbr\u003eConex\u00f5es: 4","TUV\u003cbr\u003eConex\u00f5es: 2","PLW\u003cbr\u003eConex\u00f5es: 4","MSR\u003cbr\u003eConex\u00f5es: 5","IOT\u003cbr\u003eConex\u00f5es: 1","ATA\u003cbr\u003eConex\u00f5es: 2","ESH\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.7)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[],"lon":[],"marker":{"color":"red","opacity":0.8,"size":[]},"mode":"markers","name":"2019","text":[],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.9368353700000001},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,36.204824,null,35.86166,51.165691,null,35.86166,52.132633,null,35.86166,55.378051,null,35.86166,46.603354,null,35.86166,56.130366,null,35.86166,14.058324,null,35.86166,40.463667,null,35.86166,-0.789275,null,35.86166,41.87194,null,50.850346,46.603354,null,35.86166,-35.675147,null,35.86166,61.52401,null,35.86166,-25.274398,null,35.86166,23.634501,null,35.86166,15.870032,null,37.09024,56.130366,null,37.09024,23.634501,null,23.634501,37.09024,null,35.86166,35.907757,null,52.132633,51.165691,null,35.86166,12.879721,null,35.86166,23.885942,null,35.86166,51.919438,null,35.86166,50.850346,null,35.86166,4.210484,null,35.86166,23.424076,null,51.165691,51.919438,null,14.058324,37.09024,null,35.86166,21.916221,null,35.86166,63.397768,null,35.86166,46.818188,null,35.86166,39.399872,null,51.165691,46.603354,null,35.86166,1.352083,null,20.593684,37.09024,null,51.165691,47.516231,null,35.86166,56.26392,null,35.86166,20.593684,null,55.378051,46.603354,null,35.86166,-14.235004,null,35.86166,-9.189967,null,35.86166,8.537981,null,35.86166,61.92411,null,38.963745,51.165691,null,31.791702,40.463667,null,35.86166,31.046051,null,35.86166,39.074208,null,20.593684,23.424076,null,52.132633,50.850346,null,40.463667,46.603354,null,35.86166,33.223191,null,35.86166,45.943161,null,12.565679,37.09024,null,51.165691,61.92411,null,35.86166,49.817492,null,49.817492,51.165691,null,51.919438,51.165691,null,35.86166,47.516231,null,35.86166,60.472024,null,51.165691,49.817492,null,15.870032,37.09024,null,12.879721,36.204824,null,14.058324,51.165691,null,51.165691,46.818188,null,52.132633,46.603354,null,35.86166,44.016521,null,39.399872,40.463667,null,51.165691,40.463667,null,56.130366,37.09024,null,35.86166,-30.559482,null,51.165691,41.87194,null,18.735693,37.09024,null,35.86166,55.169438,null,35.86166,26.820553,null,38.963745,37.09024,null,35.86166,22.396428,null,35.86166,7.873054,null,46.603354,50.850346,null,14.058324,36.204824,null,35.86166,-1.831239,null,-18.766947,46.603354,null,35.907757,37.09024,null,51.165691,52.132633,null,35.86166,48.379433,null,45.943161,51.919438,null,53.709807,61.52401,null,35.86166,46.151241,null,50.850346,52.132633,null,35.86166,7.946527,null,35.86166,-40.900557,null,23.885942,33.223191,null,51.919438,55.169438,null,50.850346,51.165691,null,55.378051,53.41291,null,51.919438,40.463667,null,38.963745,52.132633,null,38.963745,47.411631,null,51.919438,49.817492,null,52.132633,55.378051,null,35.86166,53.41291,null,51.165691,55.378051,null,35.86166,9.081999,null,55.378051,51.165691,null,-16.290154,6.42375,null,35.86166,4.570868,null,40.463667,41.87194,null,51.165691,63.397768,null,35.907757,36.204824,null,35.86166,30.375321,null,52.132633,41.87194,null,51.165691,50.850346,null,22.396428,37.09024,null,47.162494,41.87194,null,20.593684,7.873054,null,46.603354,51.165691,null,35.86166,23.685,null,63.397768,60.472024,null,46.603354,40.463667,null,40.463667,31.791702,null,35.86166,12.565679,null,35.86166,47.162494,null,20.593684,55.378051,null,39.399872,46.603354,null,20.593684,8.619543,null,52.132633,40.463667,null,14.058324,46.603354,null,49.817492,51.919438,null,40.463667,39.399872,null,35.86166,48.019573,null,47.411631,45.943161,null,52.132633,37.09024,null,35.86166,28.0,null,38.963745,40.463667,null,-23.442503,-14.235004,null,20.593684,9.081999,null,35.86166,40.143105,null,35.86166,58.595272,null,55.378051,40.463667,null,35.86166,29.31166,null,38.963745,46.603354,null,37.09024,50.850346,null,38.963745,42.733883,null,46.603354,55.378051,null,41.87194,46.603354,null,45.943161,51.165691,null,12.565679,36.204824,null,35.86166,12.862807,null,35.86166,-38.416097,null,26.820553,38.963745,null,37.09024,-25.274398,null,36.204824,14.058324,null,14.058324,35.907757,null,46.603354,33.886917,null,63.397768,51.919438,null,35.86166,14.497401,null,51.919438,47.162494,null,35.86166,38.963745,null,18.735693,18.971187,null,49.817492,48.669026,null,51.165691,48.669026,null,35.86166,18.735693,null,61.52401,53.709807,null,52.132633,51.919438,null,38.963745,55.378051,null,35.86166,42.733883,null,15.870032,51.165691,null,38.963745,63.397768,null,46.603354,41.87194,null,61.52401,48.019573,null,39.399872,51.165691,null,36.204824,-25.274398,null,35.86166,13.794185,null,1.352083,12.879721,null,47.162494,51.919438,null,42.733883,46.603354,null,35.86166,33.886917,null,45.943161,41.87194,null,38.963745,51.919438,null,22.396428,35.86166,null,35.86166,9.748917,null,35.86166,25.354826,null,51.165691,56.26392,null,48.379433,48.669026,null,51.919438,52.132633,null,35.86166,-32.522779,null,51.919438,48.379433,null,35.907757,14.058324,null,46.603354,49.815273,null,35.86166,15.783471,null,1.352083,-0.789275,null,35.86166,26.3351,null,45.943161,46.603354,null,55.378051,41.87194,null,51.165691,47.162494,null,37.09024,51.165691,null,22.396428,36.204824,null,63.397768,56.26392,null,40.463667,51.165691,null,63.397768,52.132633,null,41.87194,51.165691,null,35.86166,22.198745,null,4.210484,23.885942,null,35.86166,33.854721,null,35.86166,9.145,null,55.378051,37.09024,null,49.817492,46.603354,null,35.86166,45.1,null,47.162494,51.165691,null,14.058324,56.130366,null,35.86166,-16.290154,null,46.603354,52.132633,null,50.850346,49.817492,null,38.963745,39.074208,null,37.09024,36.204824,null,55.378051,63.397768,null,14.058324,55.378051,null,38.963745,45.943161,null,50.850346,40.463667,null,63.397768,61.92411,null,37.09024,55.378051,null,35.86166,56.879635,null,35.86166,30.585164,null,38.963745,33.886917,null,37.09024,52.132633,null,36.204824,35.86166,null,48.669026,49.817492,null,35.86166,7.539989,null,51.919438,48.669026,null,51.165691,45.943161,null,39.399872,49.817492,null,55.378051,51.919438,null,51.919438,56.26392,null,35.86166,31.791702,null,42.733883,51.165691,null,56.130366,23.885942,null,35.86166,41.377491,null,35.907757,22.396428,null,51.165691,61.52401,null,35.907757,35.86166,null,35.86166,42.315407,null,36.204824,50.850346,null,20.593684,46.603354,null,35.86166,4.860416,null,51.165691,53.41291,null,55.169438,51.165691,null,35.86166,-11.202692,null,15.870032,36.204824,null,38.963745,41.87194,null,20.593684,-25.274398,null,35.86166,8.619543,null,51.919438,61.92411,null,35.86166,15.199999,null,51.919438,53.709807,null,51.919438,45.943161,null,41.87194,47.516231,null,55.169438,63.397768,null,46.603354,46.818188,null,22.396428,-9.189967,null,46.603354,39.399872,null,35.86166,-18.665695,null,51.165691,37.09024,null,38.963745,47.516231,null,38.963745,47.162494,null,20.593684,51.165691,null,1.352083,-25.274398,null,4.570868,37.09024,null,20.593684,56.130366,null,20.593684,-0.023559,null,12.865416,37.09024,null,31.046051,18.735693,null,14.058324,41.87194,null,4.210484,22.396428,null,52.132633,53.41291,null,12.565679,56.130366,null,46.603354,51.919438,null,38.963745,48.669026,null,56.879635,63.397768,null,35.86166,48.669026,null,35.86166,1.373333,null,15.870032,52.132633,null,38.963745,33.223191,null,48.379433,51.165691,null,35.907757,23.424076,null,20.593684,36.204824,null,52.132633,56.26392,null,52.132633,49.817492,null,56.26392,63.397768,null,35.86166,33.0,null,56.26392,60.472024,null,42.733883,52.132633,null,36.204824,37.09024,null,37.09024,25.03428,null,20.593684,52.132633,null,51.165691,33.886917,null,35.907757,1.352083,null,1.352083,35.907757,null,35.86166,-6.369028,null,14.058324,35.86166,null,51.919438,55.378051,null,13.794185,37.09024,null,35.86166,21.512583,null,15.870032,40.463667,null,52.132633,63.397768,null,35.86166,26.02751,null,50.850346,41.87194,null,35.86166,28.394857,null,52.132633,55.169438,null,20.593684,14.497401,null,35.86166,46.862496,null,37.09024,35.86166,null,51.919438,46.603354,null,39.399872,55.378051,null,1.352083,36.204824,null,7.946527,55.378051,null,35.907757,10.691803,null,48.379433,51.919438,null,52.132633,39.399872,null,46.603354,45.943161,null,35.86166,-0.023559,null,12.565679,51.165691,null,-0.789275,1.352083,null,-30.559482,-22.95764,null,38.963745,39.399872,null,4.210484,23.634501,null,51.165691,23.634501,null,35.86166,35.126413,null,35.907757,23.634501,null,44.016521,45.943161,null,52.132633,47.516231,null,20.593684,-18.665695,null,55.378051,52.132633,null,41.87194,40.463667,null,51.165691,26.820553,null,35.86166,32.427908,null,37.09024,1.352083,null,51.919438,56.879635,null,1.352083,56.130366,null,14.058324,61.52401,null,58.595272,55.169438,null,20.593684,28.394857,null,35.86166,40.069099,null,35.907757,-0.789275,null,12.565679,48.379433,null,46.603354,35.907757,null,48.379433,45.943161,null,56.879635,35.126413,null,35.86166,11.825138,null,51.165691,35.86166,null,49.815273,23.424076,null,22.396428,56.130366,null,20.593684,21.512583,null,37.09024,18.735693,null,37.09024,35.907757,null,35.907757,15.870032,null,35.907757,51.165691,null,45.1,51.165691,null,41.87194,46.818188,null,20.593684,-30.559482,null,40.463667,37.09024,null,23.634501,9.748917,null,50.850346,51.919438,null,51.919438,47.516231,null,20.593684,41.87194,null,49.817492,40.463667,null,20.593684,23.885942,null,15.783471,37.09024,null,49.817492,41.87194,null,-0.789275,36.204824,null,22.396428,41.87194,null,38.963745,30.585164,null,37.09024,-35.675147,null,43.915886,51.165691,null,51.919438,63.397768,null,14.058324,63.397768,null,63.397768,51.165691,null,20.593684,25.354826,null,58.595272,61.92411,null,56.879635,51.165691,null,51.919438,61.52401,null,-0.789275,12.879721,null,14.058324,52.132633,null,20.593684,56.26392,null,4.210484,38.963745,null,23.634501,22.396428,null,50.850346,11.825138,null,51.165691,39.074208,null,15.870032,49.817492,null,12.565679,35.86166,null,38.963745,43.915886,null,63.397768,50.850346,null,1.352083,9.145,null,41.87194,39.074208,null,22.396428,55.378051,null,39.074208,47.162494,null,20.593684,-11.6455,null,35.907757,4.210484,null,38.963745,31.046051,null,35.86166,53.709807,null,20.593684,53.41291,null,47.411631,48.669026,null,51.165691,39.399872,null,55.378051,49.817492,null,38.963745,44.016521,null,51.919438,50.850346,null,35.86166,-2.1646,null,55.378051,23.885942,null,52.132633,46.818188,null,51.165691,60.472024,null,20.593684,17.570692,null,40.463667,55.378051,null,51.165691,49.815273,null,-16.290154,52.132633,null,51.165691,36.204824,null,52.132633,23.424076,null,49.817492,-0.023559,null,48.669026,47.516231,null,23.885942,12.862807,null,12.565679,56.26392,null,46.603354,1.352083,null,48.669026,47.162494,null,39.399872,51.919438,null,53.41291,36.204824,null,55.378051,-14.235004,null,-32.522779,-14.235004,null,35.86166,-19.015438,null,51.919438,46.151241,null,52.132633,45.943161,null,55.378051,-30.559482,null,55.378051,56.26392,null,52.132633,-14.235004,null,41.87194,38.963745,null,35.86166,35.937496,null,48.669026,21.00789,null,30.375321,37.09024,null,51.919438,41.87194,null,55.169438,56.879635,null,31.791702,50.850346,null,53.41291,51.165691,null,51.165691,55.169438,null,38.963745,21.512583,null,12.565679,46.603354,null,12.879721,15.870032,null,35.86166,-1.940278,null,50.850346,55.378051,null,38.963745,42.315407,null,4.210484,15.870032,null,21.916221,46.603354,null,38.963745,40.143105,null,49.817492,47.516231,null,52.132633,23.634501,null,31.791702,39.399872,null,22.396428,51.165691,null,52.132633,61.92411,null,35.907757,-38.416097,null,20.593684,35.86166,null,55.169438,51.919438,null,52.132633,36.204824,null,4.570868,8.537981,null,37.09024,19.3133,null,38.963745,61.52401,null,45.943161,40.463667,null,40.463667,28.0,null,38.963745,30.375321,null,50.850346,48.669026,null,42.733883,45.943161,null,35.86166,18.109581,null,48.019573,35.86166,null,1.352083,4.210484,null,48.669026,51.919438,null,48.669026,51.165691,null,52.132633,56.130366,null,45.943161,48.669026,null,14.058324,4.210484,null,14.058324,-30.559482,null,47.162494,46.603354,null,52.132633,47.162494,null,52.132633,49.815273,null,37.09024,-30.559482,null,51.165691,47.411631,null,48.379433,38.963745,null,38.963745,42.708678,null,37.09024,22.396428,null,20.593684,-35.675147,null,58.595272,63.397768,null,38.963745,50.850346,null,51.165691,42.733883,null,50.850346,28.0,null,35.86166,1.650801,null,46.603354,37.09024,null,47.162494,48.669026,null,51.919438,45.1,null,37.09024,10.691803,null,35.86166,21.00789,null,12.879721,32.3078,null,14.058324,40.463667,null,55.169438,58.595272,null,41.377491,29.31166,null,20.593684,33.0,null,20.593684,40.463667,null,14.058324,50.850346,null,38.963745,34.802075,null,35.86166,15.454166,null,14.058324,46.818188,null,50.850346,-30.559482,null,40.463667,52.132633,null,18.735693,23.634501,null,35.86166,-20.348404,null,44.016521,46.818188,null,4.210484,46.603354,null,1.352083,22.396428,null,51.165691,46.151241,null,41.87194,55.378051,null,20.593684,21.916221,null,20.593684,12.238333,null,35.86166,6.42375,null,55.378051,22.396428,null,46.818188,51.165691,null,56.26392,51.165691,null,50.850346,46.818188,null,14.058324,-25.274398,null,14.058324,51.919438,null,30.375321,51.165691,null,23.634501,56.130366,null,52.132633,60.472024,null,35.907757,12.879721,null,61.52401,56.879635,null,-30.559482,-22.328474,null,38.963745,48.019573,null,-25.274398,55.378051,null,50.850346,63.397768,null,14.058324,22.396428,null,38.963745,26.3351,null,46.603354,56.26392,null,35.86166,10.691803,null,51.165691,31.791702,null,37.09024,23.885942,null,37.09024,51.919438,null,55.378051,9.945587,null,-14.235004,22.198745,null,40.069099,38.969719,null,37.09024,13.794185,null,31.791702,46.603354,null,38.963745,35.126413,null,40.463667,1.352083,null,55.169438,61.92411,null,38.963745,48.379433,null,38.963745,46.818188,null,49.817492,52.132633,null,49.815273,46.818188,null,38.963745,49.817492,null,51.165691,58.595272,null,46.603354,47.516231,null,38.963745,4.210484,null,56.130366,-25.274398,null,12.879721,20.593684,null,-14.235004,-38.416097,null,20.593684,-6.369028,null,4.210484,-1.831239,null,37.09024,6.42375,null,22.396428,52.132633,null,37.09024,-9.189967,null,50.850346,53.41291,null,46.151241,45.1,null,36.204824,35.907757,null,15.870032,50.850346,null,52.132633,11.825138,null,51.165691,38.963745,null,40.463667,14.497401,null,47.162494,47.516231,null,20.593684,1.352083,null,35.86166,15.552727,null,48.669026,46.603354,null,61.52401,48.379433,null,38.963745,56.26392,null,14.058324,23.634501,null,21.916221,37.09024,null,15.870032,53.41291,null,20.593684,-18.766947,null,-30.559482,-18.665695,null,52.132633,45.1,null,51.165691,48.379433,null,15.870032,-25.274398,null,51.919438,37.09024,null,51.919438,60.472024,null,49.817492,50.850346,null,14.058324,-0.789275,null,23.885942,23.424076,null,45.943161,50.850346,null,30.375321,52.132633,null,37.09024,14.058324,null,4.210484,41.87194,null,20.593684,26.820553,null,46.603354,49.817492,null,56.879635,58.595272,null,22.396428,20.593684,null,37.09024,9.748917,null,45.943161,55.378051,null,44.016521,41.87194,null,37.09024,-38.416097,null,37.09024,31.046051,null,35.86166,49.815273,null,31.046051,-0.789275,null,35.86166,-13.133897,null,12.565679,52.132633,null,51.165691,56.879635,null,14.058324,23.424076,null,52.132633,48.669026,null,-30.559482,-19.015438,null,14.058324,12.565679,null,47.162494,39.399872,null,12.879721,23.424076,null,37.09024,-14.235004,null,46.603354,63.397768,null,15.870032,9.145,null,40.463667,-30.559482,null,42.733883,63.397768,null,45.1,46.151241,null,18.735693,-16.290154,null,37.09024,53.41291,null,48.669026,63.397768,null,47.411631,61.52401,null,38.963745,56.130366,null,31.791702,7.539989,null,35.86166,18.971187,null,60.472024,63.397768,null,53.41291,55.378051,null,35.86166,16.5388,null,35.86166,19.85627,null,49.817492,55.378051,null,51.165691,28.0,null,51.165691,45.1,null,41.87194,51.919438,null,20.593684,51.919438,null,51.919438,46.818188,null,55.378051,50.850346,null,55.378051,47.516231,null,1.352083,15.870032,null,12.879721,35.86166,null,-18.766947,37.09024,null,35.86166,-22.328474,null,20.593684,5.152149,null,37.09024,8.537981,null,-25.274398,-40.900557,null,48.379433,58.595272,null,47.162494,45.943161,null,58.595272,56.879635,null,20.593684,63.397768,null,61.52401,40.143105,null,55.169438,56.26392,null,4.210484,-14.235004,null,22.396428,41.377491,null,55.378051,39.399872,null,22.396428,46.603354,null,51.165691,44.016521,null,1.352083,12.565679,null,20.593684,44.016521,null,50.850346,33.886917,null,55.169438,61.52401,null,44.016521,49.817492,null,38.963745,31.791702,null,49.817492,39.074208,null,23.634501,15.783471,null,46.603354,48.669026,null,36.204824,15.870032,null,4.210484,43.915886,null,38.963745,41.377491,null,38.963745,12.862807,null,20.593684,46.818188,null,1.352083,-40.900557,null,50.850346,-38.416097,null,35.86166,5.152149,null,38.963745,53.41291,null,44.016521,51.165691,null,-0.023559,-1.940278,null,22.396428,1.352083,null,48.379433,35.86166,null,20.593684,60.472024,null,55.378051,61.52401,null,35.86166,31.952162,null,40.463667,63.397768,null,35.86166,12.865416,null,15.870032,56.130366,null,41.87194,45.943161,null,4.210484,37.09024,null,-14.235004,55.169438,null,55.378051,-25.274398,null,35.86166,3.919305,null,51.919438,42.733883,null,39.399872,52.132633,null,20.593684,42.733883,null,51.165691,30.585164,null,42.315407,35.86166,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":0.5024721197266234},"lon":[104.195397,138.252924,null,104.195397,10.451526,null,104.195397,5.291266,null,104.195397,-3.435973,null,104.195397,1.888334,null,104.195397,-106.346771,null,104.195397,108.277199,null,104.195397,-3.74922,null,104.195397,113.921327,null,104.195397,12.56738,null,4.351721,1.888334,null,104.195397,-71.542969,null,104.195397,105.318756,null,104.195397,133.775136,null,104.195397,-102.552784,null,104.195397,100.992541,null,-95.712891,-106.346771,null,-95.712891,-102.552784,null,-102.552784,-95.712891,null,104.195397,127.766922,null,5.291266,10.451526,null,104.195397,121.774017,null,104.195397,45.079162,null,104.195397,19.145136,null,104.195397,4.351721,null,104.195397,101.975766,null,104.195397,53.847818,null,10.451526,19.145136,null,108.277199,-95.712891,null,104.195397,95.955974,null,104.195397,16.354896,null,104.195397,8.227512,null,104.195397,-8.224454,null,10.451526,1.888334,null,104.195397,103.819836,null,78.96288,-95.712891,null,10.451526,14.550072,null,104.195397,9.501785,null,104.195397,78.96288,null,-3.435973,1.888334,null,104.195397,-51.92528,null,104.195397,-75.015152,null,104.195397,-80.782127,null,104.195397,25.748151,null,35.243322,10.451526,null,-7.09262,-3.74922,null,104.195397,34.851612,null,104.195397,21.824312,null,78.96288,53.847818,null,5.291266,4.351721,null,-3.74922,1.888334,null,104.195397,43.679291,null,104.195397,24.96676,null,104.990963,-95.712891,null,10.451526,25.748151,null,104.195397,15.472962,null,15.472962,10.451526,null,19.145136,10.451526,null,104.195397,14.550072,null,104.195397,8.468946,null,10.451526,15.472962,null,100.992541,-95.712891,null,121.774017,138.252924,null,108.277199,10.451526,null,10.451526,8.227512,null,5.291266,1.888334,null,104.195397,21.005859,null,-8.224454,-3.74922,null,10.451526,-3.74922,null,-106.346771,-95.712891,null,104.195397,22.937506,null,10.451526,12.56738,null,-70.162651,-95.712891,null,104.195397,23.881275,null,104.195397,30.802498,null,35.243322,-95.712891,null,104.195397,114.109497,null,104.195397,80.771797,null,1.888334,4.351721,null,108.277199,138.252924,null,104.195397,-78.183406,null,46.869107,1.888334,null,127.766922,-95.712891,null,10.451526,5.291266,null,104.195397,31.16558,null,24.96676,19.145136,null,27.953389,105.318756,null,104.195397,14.995463,null,4.351721,5.291266,null,104.195397,-1.023194,null,104.195397,174.885971,null,45.079162,43.679291,null,19.145136,23.881275,null,4.351721,10.451526,null,-3.435973,-8.24389,null,19.145136,-3.74922,null,35.243322,5.291266,null,35.243322,28.369885,null,19.145136,15.472962,null,5.291266,-3.435973,null,104.195397,-8.24389,null,10.451526,-3.435973,null,104.195397,8.675277,null,-3.435973,10.451526,null,-63.588653,-66.58973,null,104.195397,-74.297333,null,-3.74922,12.56738,null,10.451526,16.354896,null,127.766922,138.252924,null,104.195397,69.345116,null,5.291266,12.56738,null,10.451526,4.351721,null,114.109497,-95.712891,null,19.503304,12.56738,null,78.96288,80.771797,null,1.888334,10.451526,null,104.195397,90.3563,null,16.354896,8.468946,null,1.888334,-3.74922,null,-3.74922,-7.09262,null,104.195397,104.990963,null,104.195397,19.503304,null,78.96288,-3.435973,null,-8.224454,1.888334,null,78.96288,0.824782,null,5.291266,-3.74922,null,108.277199,1.888334,null,15.472962,19.145136,null,-3.74922,-8.224454,null,104.195397,66.923684,null,28.369885,24.96676,null,5.291266,-95.712891,null,104.195397,3.0,null,35.243322,-3.74922,null,-58.443832,-51.92528,null,78.96288,8.675277,null,104.195397,47.576927,null,104.195397,25.013607,null,-3.435973,-3.74922,null,104.195397,47.481766,null,35.243322,1.888334,null,-95.712891,4.351721,null,35.243322,25.48583,null,1.888334,-3.435973,null,12.56738,1.888334,null,24.96676,10.451526,null,104.990963,138.252924,null,104.195397,30.217636,null,104.195397,-63.616672,null,30.802498,35.243322,null,-95.712891,133.775136,null,138.252924,108.277199,null,108.277199,127.766922,null,1.888334,9.537499,null,16.354896,19.145136,null,104.195397,-14.452362,null,19.145136,19.503304,null,104.195397,35.243322,null,-70.162651,-72.285215,null,15.472962,19.699024,null,10.451526,19.699024,null,104.195397,-70.162651,null,105.318756,27.953389,null,5.291266,19.145136,null,35.243322,-3.435973,null,104.195397,25.48583,null,100.992541,10.451526,null,35.243322,16.354896,null,1.888334,12.56738,null,105.318756,66.923684,null,-8.224454,10.451526,null,138.252924,133.775136,null,104.195397,-88.89653,null,103.819836,121.774017,null,19.503304,19.145136,null,25.48583,1.888334,null,104.195397,9.537499,null,24.96676,12.56738,null,35.243322,19.145136,null,114.109497,104.195397,null,104.195397,-83.753428,null,104.195397,51.183884,null,10.451526,9.501785,null,31.16558,19.699024,null,19.145136,5.291266,null,104.195397,-55.765835,null,19.145136,31.16558,null,127.766922,108.277199,null,1.888334,6.129583,null,104.195397,-90.230759,null,103.819836,113.921327,null,104.195397,17.228331,null,24.96676,1.888334,null,-3.435973,12.56738,null,10.451526,19.503304,null,-95.712891,10.451526,null,114.109497,138.252924,null,16.354896,9.501785,null,-3.74922,10.451526,null,16.354896,5.291266,null,12.56738,10.451526,null,104.195397,113.543873,null,101.975766,45.079162,null,104.195397,35.862285,null,104.195397,40.489673,null,-3.435973,-95.712891,null,15.472962,1.888334,null,104.195397,15.2,null,19.503304,10.451526,null,108.277199,-106.346771,null,104.195397,-63.588653,null,1.888334,5.291266,null,4.351721,15.472962,null,35.243322,21.824312,null,-95.712891,138.252924,null,-3.435973,16.354896,null,108.277199,-3.435973,null,35.243322,24.96676,null,4.351721,-3.74922,null,16.354896,25.748151,null,-95.712891,-3.435973,null,104.195397,24.603189,null,104.195397,36.238414,null,35.243322,9.537499,null,-95.712891,5.291266,null,138.252924,104.195397,null,19.699024,15.472962,null,104.195397,-5.54708,null,19.145136,19.699024,null,10.451526,24.96676,null,-8.224454,15.472962,null,-3.435973,19.145136,null,19.145136,9.501785,null,104.195397,-7.09262,null,25.48583,10.451526,null,-106.346771,45.079162,null,104.195397,64.585262,null,127.766922,114.109497,null,10.451526,105.318756,null,127.766922,104.195397,null,104.195397,43.356892,null,138.252924,4.351721,null,78.96288,1.888334,null,104.195397,-58.93018,null,10.451526,-8.24389,null,23.881275,10.451526,null,104.195397,17.873887,null,100.992541,138.252924,null,35.243322,12.56738,null,78.96288,133.775136,null,104.195397,0.824782,null,19.145136,25.748151,null,104.195397,-86.241905,null,19.145136,27.953389,null,19.145136,24.96676,null,12.56738,14.550072,null,23.881275,16.354896,null,1.888334,8.227512,null,114.109497,-75.015152,null,1.888334,-8.224454,null,104.195397,35.529562,null,10.451526,-95.712891,null,35.243322,14.550072,null,35.243322,19.503304,null,78.96288,10.451526,null,103.819836,133.775136,null,-74.297333,-95.712891,null,78.96288,-106.346771,null,78.96288,37.906193,null,-85.207229,-95.712891,null,34.851612,-70.162651,null,108.277199,12.56738,null,101.975766,114.109497,null,5.291266,-8.24389,null,104.990963,-106.346771,null,1.888334,19.145136,null,35.243322,19.699024,null,24.603189,16.354896,null,104.195397,19.699024,null,104.195397,32.290275,null,100.992541,5.291266,null,35.243322,43.679291,null,31.16558,10.451526,null,127.766922,53.847818,null,78.96288,138.252924,null,5.291266,9.501785,null,5.291266,15.472962,null,9.501785,16.354896,null,104.195397,65.0,null,9.501785,8.468946,null,25.48583,5.291266,null,138.252924,-95.712891,null,-95.712891,-77.39628,null,78.96288,5.291266,null,10.451526,9.537499,null,127.766922,103.819836,null,103.819836,127.766922,null,104.195397,34.888822,null,108.277199,104.195397,null,19.145136,-3.435973,null,-88.89653,-95.712891,null,104.195397,55.923255,null,100.992541,-3.74922,null,5.291266,16.354896,null,104.195397,50.55096,null,4.351721,12.56738,null,104.195397,84.124008,null,5.291266,23.881275,null,78.96288,-14.452362,null,104.195397,103.846656,null,-95.712891,104.195397,null,19.145136,1.888334,null,-8.224454,-3.435973,null,103.819836,138.252924,null,-1.023194,-3.435973,null,127.766922,-61.222503,null,31.16558,19.145136,null,5.291266,-8.224454,null,1.888334,24.96676,null,104.195397,37.906193,null,104.990963,10.451526,null,113.921327,103.819836,null,22.937506,18.49041,null,35.243322,-8.224454,null,101.975766,-102.552784,null,10.451526,-102.552784,null,104.195397,33.429859,null,127.766922,-102.552784,null,21.005859,24.96676,null,5.291266,14.550072,null,78.96288,35.529562,null,-3.435973,5.291266,null,12.56738,-3.74922,null,10.451526,30.802498,null,104.195397,53.688046,null,-95.712891,103.819836,null,19.145136,24.603189,null,103.819836,-106.346771,null,108.277199,105.318756,null,25.013607,23.881275,null,78.96288,84.124008,null,104.195397,45.038189,null,127.766922,113.921327,null,104.990963,31.16558,null,1.888334,127.766922,null,31.16558,24.96676,null,24.603189,33.429859,null,104.195397,42.590275,null,10.451526,104.195397,null,6.129583,53.847818,null,114.109497,-106.346771,null,78.96288,55.923255,null,-95.712891,-70.162651,null,-95.712891,127.766922,null,127.766922,100.992541,null,127.766922,10.451526,null,15.2,10.451526,null,12.56738,8.227512,null,78.96288,22.937506,null,-3.74922,-95.712891,null,-102.552784,-83.753428,null,4.351721,19.145136,null,19.145136,14.550072,null,78.96288,12.56738,null,15.472962,-3.74922,null,78.96288,45.079162,null,-90.230759,-95.712891,null,15.472962,12.56738,null,113.921327,138.252924,null,114.109497,12.56738,null,35.243322,36.238414,null,-95.712891,-71.542969,null,17.679076,10.451526,null,19.145136,16.354896,null,108.277199,16.354896,null,16.354896,10.451526,null,78.96288,51.183884,null,25.013607,25.748151,null,24.603189,10.451526,null,19.145136,105.318756,null,113.921327,121.774017,null,108.277199,5.291266,null,78.96288,9.501785,null,101.975766,35.243322,null,-102.552784,114.109497,null,4.351721,42.590275,null,10.451526,21.824312,null,100.992541,15.472962,null,104.990963,104.195397,null,35.243322,17.679076,null,16.354896,4.351721,null,103.819836,40.489673,null,12.56738,21.824312,null,114.109497,-3.435973,null,21.824312,19.503304,null,78.96288,43.3333,null,127.766922,101.975766,null,35.243322,34.851612,null,104.195397,27.953389,null,78.96288,-8.24389,null,28.369885,19.699024,null,10.451526,-8.224454,null,-3.435973,15.472962,null,35.243322,21.005859,null,19.145136,4.351721,null,104.195397,24.15536,null,-3.435973,45.079162,null,5.291266,8.227512,null,10.451526,8.468946,null,78.96288,-3.996166,null,-3.74922,-3.435973,null,10.451526,6.129583,null,-63.588653,5.291266,null,10.451526,138.252924,null,5.291266,53.847818,null,15.472962,37.906193,null,19.699024,14.550072,null,45.079162,30.217636,null,104.990963,9.501785,null,1.888334,103.819836,null,19.699024,19.503304,null,-8.224454,19.145136,null,-8.24389,138.252924,null,-3.435973,-51.92528,null,-55.765835,-51.92528,null,104.195397,29.154857,null,19.145136,14.995463,null,5.291266,24.96676,null,-3.435973,22.937506,null,-3.435973,9.501785,null,5.291266,-51.92528,null,12.56738,35.243322,null,104.195397,14.375416,null,19.699024,-10.940835,null,69.345116,-95.712891,null,19.145136,12.56738,null,23.881275,24.603189,null,-7.09262,4.351721,null,-8.24389,10.451526,null,10.451526,23.881275,null,35.243322,55.923255,null,104.990963,1.888334,null,121.774017,100.992541,null,104.195397,29.873888,null,4.351721,-3.435973,null,35.243322,43.356892,null,101.975766,100.992541,null,95.955974,1.888334,null,35.243322,47.576927,null,15.472962,14.550072,null,5.291266,-102.552784,null,-7.09262,-8.224454,null,114.109497,10.451526,null,5.291266,25.748151,null,127.766922,-63.616672,null,78.96288,104.195397,null,23.881275,19.145136,null,5.291266,138.252924,null,-74.297333,-80.782127,null,-95.712891,-81.2546,null,35.243322,105.318756,null,24.96676,-3.74922,null,-3.74922,3.0,null,35.243322,69.345116,null,4.351721,19.699024,null,25.48583,24.96676,null,104.195397,-77.297508,null,66.923684,104.195397,null,103.819836,101.975766,null,19.699024,19.145136,null,19.699024,10.451526,null,5.291266,-106.346771,null,24.96676,19.699024,null,108.277199,101.975766,null,108.277199,22.937506,null,19.503304,1.888334,null,5.291266,19.503304,null,5.291266,6.129583,null,-95.712891,22.937506,null,10.451526,28.369885,null,31.16558,35.243322,null,35.243322,19.37439,null,-95.712891,114.109497,null,78.96288,-71.542969,null,25.013607,16.354896,null,35.243322,4.351721,null,10.451526,25.48583,null,4.351721,3.0,null,104.195397,10.267895,null,1.888334,-95.712891,null,19.503304,19.699024,null,19.145136,15.2,null,-95.712891,-61.222503,null,104.195397,-10.940835,null,121.774017,-64.7505,null,108.277199,-3.74922,null,23.881275,25.013607,null,64.585262,47.481766,null,78.96288,65.0,null,78.96288,-3.74922,null,108.277199,4.351721,null,35.243322,38.996815,null,104.195397,18.732207,null,108.277199,8.227512,null,4.351721,22.937506,null,-3.74922,5.291266,null,-70.162651,-102.552784,null,104.195397,57.552152,null,21.005859,8.227512,null,101.975766,1.888334,null,103.819836,114.109497,null,10.451526,14.995463,null,12.56738,-3.435973,null,78.96288,95.955974,null,78.96288,-1.561593,null,104.195397,-66.58973,null,-3.435973,114.109497,null,8.227512,10.451526,null,9.501785,10.451526,null,4.351721,8.227512,null,108.277199,133.775136,null,108.277199,19.145136,null,69.345116,10.451526,null,-102.552784,-106.346771,null,5.291266,8.468946,null,127.766922,121.774017,null,105.318756,24.603189,null,22.937506,24.684866,null,35.243322,66.923684,null,133.775136,-3.435973,null,4.351721,16.354896,null,108.277199,114.109497,null,35.243322,17.228331,null,1.888334,9.501785,null,104.195397,-61.222503,null,10.451526,-7.09262,null,-95.712891,45.079162,null,-95.712891,19.145136,null,-3.435973,-9.696645,null,-51.92528,113.543873,null,45.038189,59.556278,null,-95.712891,-88.89653,null,-7.09262,1.888334,null,35.243322,33.429859,null,-3.74922,103.819836,null,23.881275,25.748151,null,35.243322,31.16558,null,35.243322,8.227512,null,15.472962,5.291266,null,6.129583,8.227512,null,35.243322,15.472962,null,10.451526,25.013607,null,1.888334,14.550072,null,35.243322,101.975766,null,-106.346771,133.775136,null,121.774017,78.96288,null,-51.92528,-63.616672,null,78.96288,34.888822,null,101.975766,-78.183406,null,-95.712891,-66.58973,null,114.109497,5.291266,null,-95.712891,-75.015152,null,4.351721,-8.24389,null,14.995463,15.2,null,138.252924,127.766922,null,100.992541,4.351721,null,5.291266,42.590275,null,10.451526,35.243322,null,-3.74922,-14.452362,null,19.503304,14.550072,null,78.96288,103.819836,null,104.195397,48.516388,null,19.699024,1.888334,null,105.318756,31.16558,null,35.243322,9.501785,null,108.277199,-102.552784,null,95.955974,-95.712891,null,100.992541,-8.24389,null,78.96288,46.869107,null,22.937506,35.529562,null,5.291266,15.2,null,10.451526,31.16558,null,100.992541,133.775136,null,19.145136,-95.712891,null,19.145136,8.468946,null,15.472962,4.351721,null,108.277199,113.921327,null,45.079162,53.847818,null,24.96676,4.351721,null,69.345116,5.291266,null,-95.712891,108.277199,null,101.975766,12.56738,null,78.96288,30.802498,null,1.888334,15.472962,null,24.603189,25.013607,null,114.109497,78.96288,null,-95.712891,-83.753428,null,24.96676,-3.435973,null,21.005859,12.56738,null,-95.712891,-63.616672,null,-95.712891,34.851612,null,104.195397,6.129583,null,34.851612,113.921327,null,104.195397,27.849332,null,104.990963,5.291266,null,10.451526,24.603189,null,108.277199,53.847818,null,5.291266,19.699024,null,22.937506,29.154857,null,108.277199,104.990963,null,19.503304,-8.224454,null,121.774017,53.847818,null,-95.712891,-51.92528,null,1.888334,16.354896,null,100.992541,40.489673,null,-3.74922,22.937506,null,25.48583,16.354896,null,15.2,14.995463,null,-70.162651,-63.588653,null,-95.712891,-8.24389,null,19.699024,16.354896,null,28.369885,105.318756,null,35.243322,-106.346771,null,-7.09262,-5.54708,null,104.195397,-72.285215,null,8.468946,16.354896,null,-8.24389,-3.435973,null,104.195397,-23.0418,null,104.195397,102.495496,null,15.472962,-3.435973,null,10.451526,3.0,null,10.451526,15.2,null,12.56738,19.145136,null,78.96288,19.145136,null,19.145136,8.227512,null,-3.435973,4.351721,null,-3.435973,14.550072,null,103.819836,100.992541,null,121.774017,104.195397,null,46.869107,-95.712891,null,104.195397,24.684866,null,78.96288,46.199616,null,-95.712891,-80.782127,null,133.775136,174.885971,null,31.16558,25.013607,null,19.503304,24.96676,null,25.013607,24.603189,null,78.96288,16.354896,null,105.318756,47.576927,null,23.881275,9.501785,null,101.975766,-51.92528,null,114.109497,64.585262,null,-3.435973,-8.224454,null,114.109497,1.888334,null,10.451526,21.005859,null,103.819836,104.990963,null,78.96288,21.005859,null,4.351721,9.537499,null,23.881275,105.318756,null,21.005859,15.472962,null,35.243322,-7.09262,null,15.472962,21.824312,null,-102.552784,-90.230759,null,1.888334,19.699024,null,138.252924,100.992541,null,101.975766,17.679076,null,35.243322,64.585262,null,35.243322,30.217636,null,78.96288,8.227512,null,103.819836,174.885971,null,4.351721,-63.616672,null,104.195397,46.199616,null,35.243322,-8.24389,null,21.005859,10.451526,null,37.906193,29.873888,null,114.109497,103.819836,null,31.16558,104.195397,null,78.96288,8.468946,null,-3.435973,105.318756,null,104.195397,35.233154,null,-3.74922,16.354896,null,104.195397,-85.207229,null,100.992541,-106.346771,null,12.56738,24.96676,null,101.975766,-95.712891,null,-51.92528,23.881275,null,-3.435973,133.775136,null,104.195397,-56.027783,null,19.145136,25.48583,null,-8.224454,5.291266,null,78.96288,25.48583,null,10.451526,36.238414,null,43.356892,104.195397,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,40.463667,-11.202692,46.603354,-18.665695,55.378051,17.060816,37.09024,40.143105,31.046051,-38.416097,-35.675147,-23.442503,-32.522779,-25.274398,-9.64571,4.535277,56.130366,35.86166,-17.713371,61.92411,51.165691,7.946527,39.074208,22.396428,-0.789275,19.85627,17.570692,52.132633,-15.376706,-40.900557,8.537981,-6.314993,45.943161,1.352083,-30.559482,63.397768,15.870032,23.424076,26.820553,40.069099,53.709807,61.52401,13.193887,18.109581,10.691803,50.850346,28.0,47.516231,26.02751,43.915886,-14.235004,42.733883,7.369722,16.5388,4.570868,-0.228021,-2.1646,45.1,35.126413,49.817492,9.30769,56.26392,58.595272,61.892635,-0.803689,36.140751,18.971187,15.199999,47.162494,64.963051,53.41291,41.87194,7.539989,36.204824,30.585164,35.907757,33.854721,56.879635,55.169438,49.815273,4.210484,35.937496,-20.348404,23.634501,31.791702,12.20189,-20.904305,60.472024,12.879721,51.919438,39.399872,14.497401,44.016521,48.669026,14.058324,46.151241,12.862807,46.818188,33.886917,38.963745,1.373333,48.379433,41.608635,12.238333,32.3078,-16.290154,42.708678,-22.328474,21.521757,18.735693,-1.831239,15.783471,12.865416,7.131474,-9.189967,6.42375,42.315407,21.916221,48.019573,12.565679,13.794185,4.860416,33.223191,29.31166,25.354826,46.946947,23.885942,20.593684,3.919305,9.748917,33.0,41.0,25.03428,23.685,17.189877,-3.373056,6.611111,7.873054,-11.6455,9.145,-17.679742,11.825138,13.443182,31.952162,12.262776,9.945587,32.427908,-0.023559,40.339852,41.20438,6.428055,26.3351,22.198745,-18.766947,-13.254308,3.202778,21.00789,46.862496,47.411631,21.512583,-22.95764,28.394857,12.52111,17.607789,9.081999,30.375321,-8.874217,-1.940278,13.909444,43.94236,-4.679574,8.460555,5.152149,-19.015438,7.862684,34.802075,8.619543,38.969719,-6.369028,41.377491,-13.759029,15.552727,-13.133897,71.706936,-10.447525,-3.370417,-7.109535,13.444304,18.04248,17.897476,21.694025,-13.768752,27.514162,15.414999,12.16957,-26.522503,38.861034,7.425554,-21.236736,7.51498,-29.609988,1.650801,-19.054445,11.803749,0.18636,17.664332,19.3133,17.357822,12.984305,-15.965,16.742498,18.218785,-21.178986,15.454166,-49.280366,-90.0,-14.28522,18.420695,-0.522778,-29.040835,15.179384,41.902916,-9.2,-51.796253,-12.164165,-53.08181],"lon":[1.601554,-3.74922,17.873887,1.888334,35.529562,-3.435973,-61.796428,-95.712891,47.576927,34.851612,-63.616672,-71.542969,-58.443832,-55.765835,133.775136,160.156194,114.727669,-106.346771,104.195397,178.065033,25.748151,10.451526,-1.023194,21.824312,114.109497,113.921327,102.495496,-3.996166,5.291266,166.959158,174.885971,-80.782127,143.95555,24.96676,103.819836,22.937506,16.354896,100.992541,53.847818,30.802498,45.038189,27.953389,105.318756,-59.543198,-77.297508,-61.222503,4.351721,3.0,14.550072,50.55096,17.679076,-51.92528,25.48583,12.354722,-23.0418,-74.297333,15.827659,24.15536,15.2,33.429859,15.472962,2.315834,9.501785,25.013607,-6.911806,11.609444,-5.353585,-72.285215,-86.241905,19.503304,-19.020835,-8.24389,12.56738,-5.54708,138.252924,36.238414,127.766922,35.862285,24.603189,23.881275,6.129583,101.975766,14.375416,57.552152,-102.552784,-7.09262,-68.262383,165.618042,8.468946,121.774017,19.145136,-8.224454,-14.452362,21.005859,19.699024,108.277199,14.995463,30.217636,8.227512,9.537499,35.243322,32.290275,31.16558,21.745275,-1.561593,-64.7505,-63.588653,19.37439,24.684866,-77.781167,-70.162651,-78.183406,-90.230759,-85.207229,171.184478,-75.015152,-66.58973,43.356892,95.955974,66.923684,104.990963,-88.89653,-58.93018,43.679291,47.481766,51.183884,-56.32509,45.079162,78.96288,-56.027783,-83.753428,65.0,20.0,-77.39628,90.3563,-88.49765,29.918886,20.939444,80.771797,43.3333,40.489673,-149.406843,42.590275,-15.310139,35.233154,-61.604171,-9.696645,53.688046,37.906193,127.510093,74.766098,-9.429499,17.228331,113.543873,46.869107,34.301525,73.22068,-10.940835,103.846656,28.369885,55.923255,18.49041,84.124008,-69.968338,8.081666,8.675277,69.345116,125.727,29.873888,-60.978893,12.457777,55.491977,-11.779889,46.199616,29.154857,30.217636,38.996815,0.824782,59.556278,34.888822,64.585262,-172.104629,48.516388,27.849332,-42.604303,105.690449,-168.734039,179.194167,144.793731,-63.05483,-62.83055,-71.797928,-177.156097,90.433601,-61.370976,-68.990021,31.465866,71.276093,150.550812,-159.777671,134.58252,28.233608,10.267895,-169.867233,-15.180413,6.613081,145.94351,-81.2546,-62.782998,-61.287228,-5.7089,-62.187366,-63.043653,-175.198242,18.732207,69.348557,0.0,-170.70444,-64.639968,166.931503,167.954712,39.782334,12.453389,-171.833333,-59.523613,96.870956,73.504158],"marker":{"color":"red","opacity":0.8,"size":[8.0,20.0,20.0,20.0,16.0,20.0,8.4,20.0,20.0,20.0,20.0,20.0,18.0,19.6,20.0,4.0,16.4,20.0,20.0,15.2,20.0,20.0,18.8,20.0,20.0,20.0,12.0,11.2,20.0,4.0,20.0,19.2,10.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,17.2,20.0,20.0,18.4,10.4,20.0,20.0,13.2,20.0,14.0,20.0,20.0,20.0,11.2,4.8,20.0,10.8,15.2,20.0,20.0,20.0,8.8,20.0,20.0,6.4,8.8,8.8,8.0,11.6,20.0,20.0,20.0,20.0,13.2,20.0,13.6,20.0,16.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,2.8,8.8,20.0,20.0,20.0,20.0,19.2,20.0,20.0,20.0,20.0,10.0,20.0,13.2,20.0,14.4,20.0,20.0,11.2,9.6,17.2,17.2,8.4,8.4,20.0,20.0,20.0,19.2,5.6,20.0,11.6,20.0,16.8,20.0,20.0,20.0,11.6,13.6,17.6,18.0,1.2,20.0,20.0,8.0,20.0,9.2,13.6,9.6,13.6,6.0,6.0,6.4,11.6,3.6,12.8,9.6,8.4,6.8,4.4,7.2,7.6,7.6,20.0,0.8,15.2,8.4,10.4,14.4,20.0,10.0,10.4,8.4,16.0,20.0,14.0,20.0,8.4,8.0,7.2,14.8,20.0,2.8,12.0,5.2,5.6,10.4,6.0,6.4,11.6,6.4,3.2,10.8,7.6,19.2,19.6,3.2,6.4,16.0,5.6,0.8,2.0,1.2,4.8,4.0,1.6,4.0,1.2,3.2,4.8,9.2,3.6,6.8,3.2,2.0,2.4,6.0,8.4,0.8,4.0,2.4,2.0,7.6,3.6,4.8,2.0,1.2,2.8,2.8,6.0,0.8,1.2,1.6,4.4,1.6,0.8,2.4,0.4,0.4,0.4,0.4,0.4]},"mode":"markers","name":"2020","text":["AND\u003cbr\u003eConex\u00f5es: 20","ESP\u003cbr\u003eConex\u00f5es: 224","AGO\u003cbr\u003eConex\u00f5es: 51","FRA\u003cbr\u003eConex\u00f5es: 258","MOZ\u003cbr\u003eConex\u00f5es: 40","GBR\u003cbr\u003eConex\u00f5es: 245","ATG\u003cbr\u003eConex\u00f5es: 21","USA\u003cbr\u003eConex\u00f5es: 266","AZE\u003cbr\u003eConex\u00f5es: 51","ISR\u003cbr\u003eConex\u00f5es: 132","ARG\u003cbr\u003eConex\u00f5es: 50","CHL\u003cbr\u003eConex\u00f5es: 87","PRY\u003cbr\u003eConex\u00f5es: 45","URY\u003cbr\u003eConex\u00f5es: 49","AUS\u003cbr\u003eConex\u00f5es: 153","SLB\u003cbr\u003eConex\u00f5es: 10","BRN\u003cbr\u003eConex\u00f5es: 41","CAN\u003cbr\u003eConex\u00f5es: 190","CHN\u003cbr\u003eConex\u00f5es: 308","FJI\u003cbr\u003eConex\u00f5es: 38","FIN\u003cbr\u003eConex\u00f5es: 147","DEU\u003cbr\u003eConex\u00f5es: 275","GHA\u003cbr\u003eConex\u00f5es: 47","GRC\u003cbr\u003eConex\u00f5es: 142","HKG\u003cbr\u003eConex\u00f5es: 263","IDN\u003cbr\u003eConex\u00f5es: 139","LAO\u003cbr\u003eConex\u00f5es: 30","MLI\u003cbr\u003eConex\u00f5es: 28","NLD\u003cbr\u003eConex\u00f5es: 249","VUT\u003cbr\u003eConex\u00f5es: 10","NZL\u003cbr\u003eConex\u00f5es: 109","PAN\u003cbr\u003eConex\u00f5es: 48","PNG\u003cbr\u003eConex\u00f5es: 25","ROU\u003cbr\u003eConex\u00f5es: 126","SGP\u003cbr\u003eConex\u00f5es: 213","ZAF\u003cbr\u003eConex\u00f5es: 165","SWE\u003cbr\u003eConex\u00f5es: 175","THA\u003cbr\u003eConex\u00f5es: 178","ARE\u003cbr\u003eConex\u00f5es: 72","EGY\u003cbr\u003eConex\u00f5es: 89","ARM\u003cbr\u003eConex\u00f5es: 43","BLR\u003cbr\u003eConex\u00f5es: 80","RUS\u003cbr\u003eConex\u00f5es: 146","BRB\u003cbr\u003eConex\u00f5es: 46","JAM\u003cbr\u003eConex\u00f5es: 26","TTO\u003cbr\u003eConex\u00f5es: 56","BEL\u003cbr\u003eConex\u00f5es: 229","DZA\u003cbr\u003eConex\u00f5es: 33","AUT\u003cbr\u003eConex\u00f5es: 59","BHR\u003cbr\u003eConex\u00f5es: 35","BIH\u003cbr\u003eConex\u00f5es: 70","BRA\u003cbr\u003eConex\u00f5es: 151","BGR\u003cbr\u003eConex\u00f5es: 133","CMR\u003cbr\u003eConex\u00f5es: 28","CPV\u003cbr\u003eConex\u00f5es: 12","COL\u003cbr\u003eConex\u00f5es: 103","COG\u003cbr\u003eConex\u00f5es: 27","COD\u003cbr\u003eConex\u00f5es: 38","HRV\u003cbr\u003eConex\u00f5es: 103","CYP\u003cbr\u003eConex\u00f5es: 66","CZE\u003cbr\u003eConex\u00f5es: 181","BEN\u003cbr\u003eConex\u00f5es: 22","DNK\u003cbr\u003eConex\u00f5es: 177","EST\u003cbr\u003eConex\u00f5es: 97","FRO\u003cbr\u003eConex\u00f5es: 16","GAB\u003cbr\u003eConex\u00f5es: 22","GIB\u003cbr\u003eConex\u00f5es: 22","HTI\u003cbr\u003eConex\u00f5es: 20","HND\u003cbr\u003eConex\u00f5es: 29","HUN\u003cbr\u003eConex\u00f5es: 141","ISL\u003cbr\u003eConex\u00f5es: 50","IRL\u003cbr\u003eConex\u00f5es: 153","ITA\u003cbr\u003eConex\u00f5es: 215","CIV\u003cbr\u003eConex\u00f5es: 33","JPN\u003cbr\u003eConex\u00f5es: 161","JOR\u003cbr\u003eConex\u00f5es: 34","KOR\u003cbr\u003eConex\u00f5es: 224","LBN\u003cbr\u003eConex\u00f5es: 40","LVA\u003cbr\u003eConex\u00f5es: 104","LTU\u003cbr\u003eConex\u00f5es: 129","LUX\u003cbr\u003eConex\u00f5es: 122","MYS\u003cbr\u003eConex\u00f5es: 127","MLT\u003cbr\u003eConex\u00f5es: 71","MUS\u003cbr\u003eConex\u00f5es: 57","MEX\u003cbr\u003eConex\u00f5es: 93","MAR\u003cbr\u003eConex\u00f5es: 109","BES\u003cbr\u003eConex\u00f5es: 7","NCL\u003cbr\u003eConex\u00f5es: 22","NOR\u003cbr\u003eConex\u00f5es: 119","PHL\u003cbr\u003eConex\u00f5es: 115","POL\u003cbr\u003eConex\u00f5es: 206","PRT\u003cbr\u003eConex\u00f5es: 199","SEN\u003cbr\u003eConex\u00f5es: 48","SRB\u003cbr\u003eConex\u00f5es: 100","SVK\u003cbr\u003eConex\u00f5es: 126","VNM\u003cbr\u003eConex\u00f5es: 141","SVN\u003cbr\u003eConex\u00f5es: 124","SDN\u003cbr\u003eConex\u00f5es: 25","CHE\u003cbr\u003eConex\u00f5es: 220","TUN\u003cbr\u003eConex\u00f5es: 33","TUR\u003cbr\u003eConex\u00f5es: 228","UGA\u003cbr\u003eConex\u00f5es: 36","UKR\u003cbr\u003eConex\u00f5es: 134","MKD\u003cbr\u003eConex\u00f5es: 65","BFA\u003cbr\u003eConex\u00f5es: 28","BMU\u003cbr\u003eConex\u00f5es: 24","BOL\u003cbr\u003eConex\u00f5es: 43","MNE\u003cbr\u003eConex\u00f5es: 43","BWA\u003cbr\u003eConex\u00f5es: 21","CUB\u003cbr\u003eConex\u00f5es: 21","DOM\u003cbr\u003eConex\u00f5es: 80","ECU\u003cbr\u003eConex\u00f5es: 52","GTM\u003cbr\u003eConex\u00f5es: 67","NIC\u003cbr\u003eConex\u00f5es: 48","MHL\u003cbr\u003eConex\u00f5es: 14","PER\u003cbr\u003eConex\u00f5es: 78","VEN\u003cbr\u003eConex\u00f5es: 29","GEO\u003cbr\u003eConex\u00f5es: 72","MMR\u003cbr\u003eConex\u00f5es: 42","KAZ\u003cbr\u003eConex\u00f5es: 64","KHM\u003cbr\u003eConex\u00f5es: 67","SLV\u003cbr\u003eConex\u00f5es: 53","GUY\u003cbr\u003eConex\u00f5es: 29","IRQ\u003cbr\u003eConex\u00f5es: 34","KWT\u003cbr\u003eConex\u00f5es: 44","QAT\u003cbr\u003eConex\u00f5es: 45","SPM\u003cbr\u003eConex\u00f5es: 3","SAU\u003cbr\u003eConex\u00f5es: 66","IND\u003cbr\u003eConex\u00f5es: 230","SUR\u003cbr\u003eConex\u00f5es: 20","CRI\u003cbr\u003eConex\u00f5es: 60","AFG\u003cbr\u003eConex\u00f5es: 23","ALB\u003cbr\u003eConex\u00f5es: 34","BHS\u003cbr\u003eConex\u00f5es: 24","BGD\u003cbr\u003eConex\u00f5es: 34","BLZ\u003cbr\u003eConex\u00f5es: 15","BDI\u003cbr\u003eConex\u00f5es: 15","CAF\u003cbr\u003eConex\u00f5es: 16","LKA\u003cbr\u003eConex\u00f5es: 29","COM\u003cbr\u003eConex\u00f5es: 9","ETH\u003cbr\u003eConex\u00f5es: 32","PYF\u003cbr\u003eConex\u00f5es: 24","DJI\u003cbr\u003eConex\u00f5es: 21","GMB\u003cbr\u003eConex\u00f5es: 17","PSE\u003cbr\u003eConex\u00f5es: 11","GRD\u003cbr\u003eConex\u00f5es: 18","GIN\u003cbr\u003eConex\u00f5es: 19","IRN\u003cbr\u003eConex\u00f5es: 19","KEN\u003cbr\u003eConex\u00f5es: 80","PRK\u003cbr\u003eConex\u00f5es: 2","KGZ\u003cbr\u003eConex\u00f5es: 38","LBR\u003cbr\u003eConex\u00f5es: 21","LBY\u003cbr\u003eConex\u00f5es: 26","MAC\u003cbr\u003eConex\u00f5es: 36","MDG\u003cbr\u003eConex\u00f5es: 51","MWI\u003cbr\u003eConex\u00f5es: 25","MDV\u003cbr\u003eConex\u00f5es: 26","MRT\u003cbr\u003eConex\u00f5es: 21","MNG\u003cbr\u003eConex\u00f5es: 40","MDA\u003cbr\u003eConex\u00f5es: 63","OMN\u003cbr\u003eConex\u00f5es: 35","NAM\u003cbr\u003eConex\u00f5es: 71","NPL\u003cbr\u003eConex\u00f5es: 21","ABW\u003cbr\u003eConex\u00f5es: 20","NER\u003cbr\u003eConex\u00f5es: 18","NGA\u003cbr\u003eConex\u00f5es: 37","PAK\u003cbr\u003eConex\u00f5es: 123","TLS\u003cbr\u003eConex\u00f5es: 7","RWA\u003cbr\u003eConex\u00f5es: 30","LCA\u003cbr\u003eConex\u00f5es: 13","SMR\u003cbr\u003eConex\u00f5es: 14","SYC\u003cbr\u003eConex\u00f5es: 26","SLE\u003cbr\u003eConex\u00f5es: 15","SOM\u003cbr\u003eConex\u00f5es: 16","ZWE\u003cbr\u003eConex\u00f5es: 29","SSD\u003cbr\u003eConex\u00f5es: 16","SYR\u003cbr\u003eConex\u00f5es: 8","TGO\u003cbr\u003eConex\u00f5es: 27","TKM\u003cbr\u003eConex\u00f5es: 19","TZA\u003cbr\u003eConex\u00f5es: 48","UZB\u003cbr\u003eConex\u00f5es: 49","WSM\u003cbr\u003eConex\u00f5es: 8","YEM\u003cbr\u003eConex\u00f5es: 16","ZMB\u003cbr\u003eConex\u00f5es: 40","GRL\u003cbr\u003eConex\u00f5es: 14","CXR\u003cbr\u003eConex\u00f5es: 2","KIR\u003cbr\u003eConex\u00f5es: 5","TUV\u003cbr\u003eConex\u00f5es: 3","GUM\u003cbr\u003eConex\u00f5es: 12","SXM\u003cbr\u003eConex\u00f5es: 10","BLM\u003cbr\u003eConex\u00f5es: 4","TCA\u003cbr\u003eConex\u00f5es: 10","WLF\u003cbr\u003eConex\u00f5es: 3","BTN\u003cbr\u003eConex\u00f5es: 8","DMA\u003cbr\u003eConex\u00f5es: 12","CUW\u003cbr\u003eConex\u00f5es: 23","SWZ\u003cbr\u003eConex\u00f5es: 9","TJK\u003cbr\u003eConex\u00f5es: 17","FSM\u003cbr\u003eConex\u00f5es: 8","COK\u003cbr\u003eConex\u00f5es: 5","PLW\u003cbr\u003eConex\u00f5es: 6","LSO\u003cbr\u003eConex\u00f5es: 15","GNQ\u003cbr\u003eConex\u00f5es: 21","NIU\u003cbr\u003eConex\u00f5es: 2","GNB\u003cbr\u003eConex\u00f5es: 10","STP\u003cbr\u003eConex\u00f5es: 6","MNP\u003cbr\u003eConex\u00f5es: 5","CYM\u003cbr\u003eConex\u00f5es: 19","KNA\u003cbr\u003eConex\u00f5es: 9","VCT\u003cbr\u003eConex\u00f5es: 12","SHN\u003cbr\u003eConex\u00f5es: 5","MSR\u003cbr\u003eConex\u00f5es: 3","AIA\u003cbr\u003eConex\u00f5es: 7","TON\u003cbr\u003eConex\u00f5es: 7","TCD\u003cbr\u003eConex\u00f5es: 15","ATF\u003cbr\u003eConex\u00f5es: 2","ATA\u003cbr\u003eConex\u00f5es: 3","ASM\u003cbr\u003eConex\u00f5es: 4","VGB\u003cbr\u003eConex\u00f5es: 11","NRU\u003cbr\u003eConex\u00f5es: 4","NFK\u003cbr\u003eConex\u00f5es: 2","ERI\u003cbr\u003eConex\u00f5es: 6","VAT\u003cbr\u003eConex\u00f5es: 1","TKL\u003cbr\u003eConex\u00f5es: 1","FLK\u003cbr\u003eConex\u00f5es: 1","CCK\u003cbr\u003eConex\u00f5es: 1","HMD\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.9573728699999999},"lon":[104.195397,-95.712891,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,36.204824,null,35.86166,51.165691,null,35.86166,55.378051,null,35.86166,52.132633,null,35.86166,-35.675147,null,35.86166,56.130366,null,35.86166,23.634501,null,35.86166,-0.789275,null,35.86166,46.603354,null,35.86166,15.870032,null,37.09024,23.634501,null,23.634501,37.09024,null,35.86166,-25.274398,null,51.919438,56.26392,null,35.86166,14.058324,null,52.132633,51.165691,null,35.86166,40.463667,null,35.86166,35.907757,null,35.86166,12.879721,null,37.09024,56.130366,null,35.86166,23.424076,null,35.86166,4.210484,null,14.058324,37.09024,null,35.86166,51.919438,null,35.86166,21.916221,null,38.963745,46.603354,null,35.86166,41.87194,null,35.86166,61.52401,null,35.86166,-14.235004,null,51.165691,46.603354,null,35.86166,-9.189967,null,35.86166,50.850346,null,35.86166,-30.559482,null,52.132633,46.603354,null,20.593684,23.424076,null,51.919438,51.165691,null,38.963745,51.165691,null,51.165691,51.919438,null,12.565679,37.09024,null,20.593684,37.09024,null,35.86166,4.570868,null,35.86166,63.397768,null,51.165691,47.516231,null,38.963745,37.09024,null,4.210484,46.603354,null,35.86166,1.352083,null,35.86166,23.885942,null,52.132633,50.850346,null,33.886917,46.603354,null,35.86166,20.593684,null,38.963745,52.132633,null,35.86166,-1.831239,null,12.879721,36.204824,null,18.735693,37.09024,null,51.165691,41.87194,null,50.850346,52.132633,null,35.86166,39.074208,null,56.130366,37.09024,null,51.165691,49.817492,null,35.86166,-40.900557,null,51.165691,46.818188,null,51.165691,52.132633,null,35.86166,8.537981,null,35.86166,56.26392,null,35.86166,46.818188,null,48.669026,49.817492,null,1.352083,-25.274398,null,51.165691,40.463667,null,35.86166,22.396428,null,31.791702,40.463667,null,39.399872,40.463667,null,35.86166,33.223191,null,42.733883,45.943161,null,35.86166,60.472024,null,35.86166,61.92411,null,38.963745,41.87194,null,37.09024,30.375321,null,35.86166,46.151241,null,35.86166,12.565679,null,55.378051,46.603354,null,14.058324,55.378051,null,52.132633,51.919438,null,46.603354,51.165691,null,35.86166,15.783471,null,35.86166,31.046051,null,40.463667,39.399872,null,35.86166,-38.416097,null,15.870032,37.09024,null,40.463667,46.603354,null,14.058324,36.204824,null,37.09024,-25.274398,null,63.397768,60.472024,null,49.817492,51.165691,null,38.963745,39.074208,null,38.963745,47.411631,null,38.963745,40.463667,null,51.165691,50.850346,null,46.603354,33.886917,null,42.315407,51.919438,null,20.593684,7.873054,null,51.919438,49.817492,null,35.86166,26.820553,null,53.709807,61.52401,null,-18.766947,46.603354,null,45.943161,51.919438,null,40.463667,41.87194,null,46.603354,40.463667,null,35.86166,48.379433,null,39.399872,46.603354,null,35.86166,13.794185,null,20.593684,9.081999,null,35.86166,48.019573,null,35.86166,53.41291,null,14.058324,51.165691,null,55.378051,40.463667,null,35.907757,37.09024,null,35.86166,28.0,null,35.86166,9.748917,null,49.817492,37.09024,null,52.132633,41.87194,null,35.86166,45.943161,null,14.058324,35.907757,null,46.603354,50.850346,null,35.86166,39.399872,null,46.603354,41.87194,null,35.86166,9.081999,null,51.919438,52.132633,null,41.87194,46.603354,null,52.132633,37.09024,null,35.86166,49.817492,null,22.396428,35.86166,null,35.86166,7.946527,null,35.907757,36.204824,null,-23.442503,-14.235004,null,51.165691,63.397768,null,35.86166,18.735693,null,35.86166,47.516231,null,46.603354,52.132633,null,63.397768,56.26392,null,55.378051,53.41291,null,55.378051,39.399872,null,35.907757,22.396428,null,35.86166,-32.522779,null,35.907757,7.946527,null,37.09024,50.850346,null,63.397768,51.919438,null,38.963745,55.378051,null,51.165691,55.378051,null,35.86166,47.162494,null,38.963745,39.399872,null,20.593684,51.165691,null,42.733883,52.132633,null,35.86166,26.3351,null,45.943161,46.603354,null,35.86166,23.685,null,37.09024,23.685,null,49.817492,51.919438,null,51.919438,47.162494,null,61.52401,53.709807,null,55.378051,51.165691,null,35.86166,55.169438,null,4.210484,38.963745,null,40.463667,31.791702,null,22.396428,37.09024,null,20.593684,-6.369028,null,20.593684,52.132633,null,45.943161,51.165691,null,33.886917,41.87194,null,20.593684,28.394857,null,14.058324,52.132633,null,1.352083,-40.900557,null,51.919438,46.603354,null,47.411631,45.943161,null,50.850346,46.603354,null,35.86166,25.354826,null,51.165691,56.26392,null,35.907757,14.058324,null,15.870032,51.165691,null,37.09024,55.378051,null,38.963745,45.943161,null,63.397768,61.92411,null,52.132633,40.463667,null,35.86166,32.427908,null,38.963745,42.733883,null,52.132633,55.378051,null,35.86166,15.199999,null,39.074208,47.162494,null,38.963745,26.3351,null,37.09024,51.165691,null,55.378051,52.132633,null,61.52401,48.019573,null,15.870032,49.817492,null,21.916221,35.907757,null,37.09024,7.873054,null,50.850346,51.165691,null,20.593684,55.378051,null,49.817492,48.669026,null,35.86166,29.31166,null,56.26392,60.472024,null,33.886917,55.378051,null,51.919438,48.379433,null,38.963745,61.92411,null,51.919438,41.87194,null,41.87194,51.165691,null,35.86166,42.733883,null,35.86166,38.963745,null,35.86166,-20.348404,null,31.046051,-1.831239,null,35.907757,51.165691,null,30.375321,37.09024,null,37.09024,8.460555,null,35.86166,33.854721,null,52.132633,53.41291,null,35.86166,30.375321,null,1.352083,4.210484,null,51.165691,37.09024,null,49.817492,46.603354,null,38.963745,47.516231,null,53.709807,52.132633,null,35.86166,48.669026,null,56.879635,63.397768,null,55.169438,51.165691,null,38.963745,63.397768,null,36.204824,14.058324,null,50.850346,49.817492,null,15.870032,52.132633,null,51.165691,47.162494,null,35.86166,1.373333,null,40.463667,51.165691,null,35.86166,-6.369028,null,35.86166,30.585164,null,52.132633,38.963745,null,35.86166,4.860416,null,35.86166,-0.023559,null,12.565679,56.130366,null,35.86166,6.42375,null,46.603354,45.943161,null,47.162494,51.165691,null,46.603354,39.399872,null,38.963745,47.162494,null,45.943161,41.87194,null,14.058324,56.130366,null,44.016521,51.165691,null,35.86166,45.1,null,51.919438,63.397768,null,20.593684,46.603354,null,1.352083,-38.416097,null,36.204824,37.09024,null,52.132633,49.817492,null,41.87194,47.162494,null,20.593684,36.204824,null,35.86166,46.862496,null,52.132633,55.169438,null,38.963745,33.223191,null,37.09024,17.189877,null,51.165691,46.151241,null,35.86166,21.512583,null,35.86166,-18.766947,null,46.603354,53.41291,null,35.907757,1.352083,null,46.603354,46.818188,null,48.379433,48.669026,null,51.919438,45.943161,null,42.315407,52.132633,null,12.565679,36.204824,null,20.593684,41.87194,null,33.886917,37.09024,null,38.963745,50.850346,null,51.165691,53.41291,null,38.963745,56.26392,null,37.09024,1.352083,null,51.165691,45.943161,null,14.058324,-25.274398,null,36.204824,-25.274398,null,4.210484,22.396428,null,51.165691,61.92411,null,38.963745,29.31166,null,51.919438,48.669026,null,1.352083,15.870032,null,49.817492,39.399872,null,51.165691,48.669026,null,47.162494,55.169438,null,35.907757,35.86166,null,35.86166,-16.290154,null,55.169438,63.397768,null,15.870032,36.204824,null,20.593684,8.619543,null,42.733883,51.165691,null,20.593684,-0.023559,null,46.603354,55.378051,null,51.919438,55.169438,null,35.86166,-11.202692,null,44.016521,45.943161,null,50.850346,40.463667,null,37.09024,35.86166,null,38.963745,49.817492,null,48.379433,45.943161,null,51.165691,23.634501,null,39.074208,41.87194,null,51.919438,47.516231,null,51.165691,49.815273,null,20.593684,56.130366,null,14.058324,35.86166,null,35.86166,42.315407,null,14.058324,63.397768,null,35.907757,-0.789275,null,-1.831239,4.570868,null,51.919438,61.92411,null,41.87194,60.472024,null,7.946527,55.378051,null,7.946527,51.165691,null,63.397768,52.132633,null,23.634501,1.352083,null,42.315407,50.850346,null,52.132633,46.818188,null,48.669026,51.919438,null,50.850346,51.919438,null,-32.522779,-14.235004,null,-16.290154,6.42375,null,36.204824,35.86166,null,55.378051,37.09024,null,22.396428,12.879721,null,38.963745,31.046051,null,46.603354,63.397768,null,47.162494,52.132633,null,4.210484,15.870032,null,37.09024,36.204824,null,47.162494,48.669026,null,35.86166,40.339852,null,14.058324,40.463667,null,47.162494,41.87194,null,48.379433,51.165691,null,39.074208,40.463667,null,31.791702,39.399872,null,14.058324,22.396428,null,42.733883,49.815273,null,35.86166,35.937496,null,42.733883,39.074208,null,26.820553,22.396428,null,-25.274398,-40.900557,null,35.907757,4.210484,null,52.132633,56.26392,null,51.165691,61.52401,null,42.315407,51.165691,null,41.87194,46.818188,null,47.162494,45.943161,null,51.919438,40.463667,null,48.669026,47.516231,null,38.963745,51.919438,null,55.378051,50.850346,null,51.165691,39.399872,null,35.86166,31.791702,null,51.919438,50.850346,null,46.603354,51.919438,null,1.352083,-1.831239,null,51.165691,42.733883,null,51.165691,-25.274398,null,35.907757,9.945587,null,35.86166,26.02751,null,20.593684,-25.274398,null,51.165691,55.169438,null,26.820553,38.963745,null,56.26392,63.397768,null,38.963745,55.169438,null,20.593684,15.870032,null,42.733883,41.87194,null,14.058324,51.919438,null,35.86166,7.873054,null,39.399872,48.669026,null,4.210484,1.352083,null,51.165691,35.86166,null,35.86166,14.497401,null,35.86166,40.143105,null,51.165691,-30.559482,null,40.463667,55.378051,null,14.058324,-0.789275,null,1.352083,37.09024,null,35.86166,17.607789,null,35.86166,41.377491,null,38.963745,49.815273,null,35.86166,41.20438,null,35.86166,58.595272,null,35.907757,23.634501,null,55.378051,31.791702,null,37.09024,9.748917,null,14.058324,46.603354,null,38.963745,42.315407,null,-14.235004,-38.416097,null,49.817492,50.850346,null,12.879721,37.09024,null,38.963745,46.818188,null,41.87194,55.378051,null,37.09024,23.885942,null,46.603354,-40.900557,null,46.603354,37.09024,null,1.352083,23.634501,null,1.352083,19.85627,null,38.963745,48.669026,null,53.41291,36.204824,null,39.399872,55.378051,null,55.169438,58.595272,null,51.919438,55.378051,null,52.132633,23.634501,null,51.919438,60.472024,null,55.378051,47.516231,null,38.963745,46.151241,null,35.86166,-18.665695,null,51.165691,60.472024,null,35.86166,8.619543,null,47.411631,48.669026,null,-0.789275,12.879721,null,58.595272,61.92411,null,15.870032,-25.274398,null,51.919438,61.52401,null,35.86166,-19.015438,null,35.86166,-1.940278,null,35.907757,-25.274398,null,45.1,51.165691,null,55.169438,51.919438,null,35.86166,7.369722,null,49.817492,41.87194,null,51.165691,45.1,null,1.352083,22.396428,null,42.733883,46.603354,null,51.919438,56.879635,null,63.397768,51.165691,null,48.019573,20.593684,null,30.375321,51.165691,null,-30.559482,-22.95764,null,41.87194,39.074208,null,55.378051,51.919438,null,26.820553,9.081999,null,37.09024,52.132633,null,49.817492,47.516231,null,60.472024,63.397768,null,55.378051,23.424076,null,15.783471,15.199999,null,60.472024,41.87194,null,38.963745,44.016521,null,41.87194,51.919438,null,1.352083,-0.789275,null,47.162494,45.1,null,47.162494,39.074208,null,33.886917,50.850346,null,37.09024,-35.675147,null,-25.274398,37.09024,null,48.669026,46.603354,null,48.669026,47.162494,null,63.397768,55.378051,null,41.87194,40.463667,null,35.907757,8.537981,null,12.565679,55.378051,null,4.210484,-25.274398,null,35.86166,15.552727,null,14.058324,61.52401,null,23.634501,15.783471,null,35.86166,-17.713371,null,35.86166,11.825138,null,51.165691,39.074208,null,35.86166,56.879635,null,20.593684,21.916221,null,39.074208,48.669026,null,14.058324,41.87194,null,51.165691,23.424076,null,-0.789275,1.352083,null,35.907757,15.870032,null,-14.235004,-32.522779,null,51.165691,38.963745,null,14.058324,7.539989,null,48.379433,51.919438,null,41.377491,38.861034,null,31.046051,37.09024,null,55.378051,-25.274398,null,22.396428,14.058324,null,52.132633,60.472024,null,35.86166,35.126413,null,18.735693,41.87194,null,12.565679,35.86166,null,51.919438,37.09024,null,55.169438,56.26392,null,56.130366,18.971187,null,52.132633,63.397768,null,61.52401,48.379433,null,35.86166,-13.133897,null,46.818188,51.165691,null,39.399872,51.165691,null,38.963745,30.375321,null,23.885942,33.854721,null,38.963745,40.143105,null,42.315407,55.378051,null,45.943161,40.463667,null,35.907757,12.879721,null,35.86166,-6.314993,null,35.86166,44.016521,null,52.132633,45.943161,null,35.86166,12.865416,null,20.593684,56.26392,null,51.165691,44.016521,null,51.165691,26.820553,null,35.86166,33.886917,null,38.963745,53.41291,null,35.86166,28.394857,null,50.850346,41.87194,null,48.669026,37.09024,null,51.165691,31.791702,null,42.315407,46.603354,null,14.058324,12.565679,null,49.817492,55.378051,null,56.26392,51.165691,null,56.879635,58.595272,null,35.86166,19.85627,null,35.86166,53.709807,null,37.09024,-30.559482,null,52.132633,-14.235004,null,35.86166,10.691803,null,22.396428,55.378051,null,14.058324,-30.559482,null,38.963745,30.585164,null,14.058324,23.424076,null,55.169438,56.879635,null,-0.023559,6.42375,null,10.691803,4.860416,null,38.963745,28.0,null,50.850346,46.151241,null,37.09024,4.210484,null,61.92411,51.165691,null,47.162494,46.603354,null,-30.559482,-19.015438,null,20.593684,9.30769,null,31.791702,55.378051,null,35.86166,-2.1646,null,35.86166,5.152149,null,14.058324,23.685,null,38.963745,31.791702,null,45.943161,42.733883,null,20.593684,4.210484,null,45.1,46.151241,null,22.396428,22.198745,null,50.850346,49.815273,null,-14.235004,37.09024,null,52.132633,39.074208,null,1.352083,12.879721,null,41.87194,45.943161,null,31.791702,51.165691,null,35.86166,-20.904305,null,38.963745,35.126413,null,61.92411,63.397768,null,30.375321,50.850346,null,30.375321,52.132633,null,56.26392,46.603354,null,39.399872,50.850346,null,51.919438,46.151241,null,1.352083,36.204824,null,38.963745,-25.274398,null,35.86166,18.109581,null,33.886917,48.669026,null,35.86166,9.145,null,20.593684,-30.559482,null,38.963745,43.915886,null,52.132633,47.162494,null,35.86166,4.535277,null,36.204824,35.907757,null,43.915886,51.165691,null,-0.789275,36.204824,null,22.396428,51.165691,null,52.132633,47.516231,null,37.09024,10.691803,null,35.86166,9.945587,null,50.850346,37.09024,null,51.919438,39.074208,null,30.375321,55.378051,null,46.151241,45.1,null,40.463667,52.132633,null,45.943161,50.850346,null,51.165691,48.379433,null,42.733883,50.850346,null,45.943161,52.132633,null,41.87194,41.0,null,40.463667,35.86166,null,37.09024,23.424076,null,26.02751,37.09024,null,-30.559482,-22.328474,null,20.593684,33.0,null,36.204824,22.396428,null,35.907757,56.130366,null,14.058324,15.870032,null,38.963745,41.608635,null,20.593684,23.685,null,14.058324,4.210484,null,48.379433,38.963745,null,38.963745,45.1,null,45.943161,46.818188,null,20.593684,8.537981,null,45.943161,55.378051,null,12.865416,18.735693,null,1.352083,9.748917,null,52.132633,61.92411,null,22.396428,36.204824,null,33.886917,30.585164,null,39.399872,51.919438,null,20.593684,39.399872,null,37.09024,15.870032,null,44.016521,52.132633,null,-0.023559,-1.940278,null,55.169438,61.52401,null,48.669026,51.165691,null,46.603354,35.907757,null,35.86166,7.539989,null,37.09024,8.537981,null,56.26392,23.685,null,58.595272,63.397768,null,12.565679,51.165691,null,45.943161,48.669026,null,52.132633,48.669026,null,4.570868,35.86166,null,53.41291,55.378051,null,14.058324,20.593684,null,51.165691,23.885942,null,53.709807,-9.189967,null,35.907757,17.189877,null,56.879635,53.709807,null,37.09024,18.735693,null,42.733883,23.634501,null,49.817492,56.130366,null,20.593684,30.375321,null,14.058324,45.943161,null,35.907757,45.943161,null,33.886917,40.463667,null,15.870032,32.427908,null,20.593684,14.497401,null,52.132633,44.016521,null,36.204824,-0.789275,null,56.26392,61.52401,null,4.210484,20.593684,null,46.603354,55.169438,null,4.210484,36.204824,null,63.397768,64.963051,null,51.919438,53.709807,null,56.130366,50.850346,null,1.352083,35.907757,null,50.850346,33.886917,null,14.058324,50.850346,null,13.794185,12.865416,null,-0.789275,-4.679574,null,35.86166,38.969719,null,35.86166,3.202778,null,55.378051,36.204824,null,23.634501,56.130366,null,38.963745,17.570692,null,45.943161,39.074208,null,35.907757,20.593684,null,38.963745,7.946527,null,46.603354,-17.679742,null,47.162494,40.463667,null,52.132633,56.130366,null,35.86166,22.198745,null,46.151241,51.165691,null,37.09024,38.963745,null,56.879635,55.169438,null,40.463667,63.397768,null,52.132633,64.963051,null,47.411631,61.52401,null,51.165691,33.886917,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":0.5023030564308597},"lon":[104.195397,138.252924,null,104.195397,10.451526,null,104.195397,-3.435973,null,104.195397,5.291266,null,104.195397,-71.542969,null,104.195397,-106.346771,null,104.195397,-102.552784,null,104.195397,113.921327,null,104.195397,1.888334,null,104.195397,100.992541,null,-95.712891,-102.552784,null,-102.552784,-95.712891,null,104.195397,133.775136,null,19.145136,9.501785,null,104.195397,108.277199,null,5.291266,10.451526,null,104.195397,-3.74922,null,104.195397,127.766922,null,104.195397,121.774017,null,-95.712891,-106.346771,null,104.195397,53.847818,null,104.195397,101.975766,null,108.277199,-95.712891,null,104.195397,19.145136,null,104.195397,95.955974,null,35.243322,1.888334,null,104.195397,12.56738,null,104.195397,105.318756,null,104.195397,-51.92528,null,10.451526,1.888334,null,104.195397,-75.015152,null,104.195397,4.351721,null,104.195397,22.937506,null,5.291266,1.888334,null,78.96288,53.847818,null,19.145136,10.451526,null,35.243322,10.451526,null,10.451526,19.145136,null,104.990963,-95.712891,null,78.96288,-95.712891,null,104.195397,-74.297333,null,104.195397,16.354896,null,10.451526,14.550072,null,35.243322,-95.712891,null,101.975766,1.888334,null,104.195397,103.819836,null,104.195397,45.079162,null,5.291266,4.351721,null,9.537499,1.888334,null,104.195397,78.96288,null,35.243322,5.291266,null,104.195397,-78.183406,null,121.774017,138.252924,null,-70.162651,-95.712891,null,10.451526,12.56738,null,4.351721,5.291266,null,104.195397,21.824312,null,-106.346771,-95.712891,null,10.451526,15.472962,null,104.195397,174.885971,null,10.451526,8.227512,null,10.451526,5.291266,null,104.195397,-80.782127,null,104.195397,9.501785,null,104.195397,8.227512,null,19.699024,15.472962,null,103.819836,133.775136,null,10.451526,-3.74922,null,104.195397,114.109497,null,-7.09262,-3.74922,null,-8.224454,-3.74922,null,104.195397,43.679291,null,25.48583,24.96676,null,104.195397,8.468946,null,104.195397,25.748151,null,35.243322,12.56738,null,-95.712891,69.345116,null,104.195397,14.995463,null,104.195397,104.990963,null,-3.435973,1.888334,null,108.277199,-3.435973,null,5.291266,19.145136,null,1.888334,10.451526,null,104.195397,-90.230759,null,104.195397,34.851612,null,-3.74922,-8.224454,null,104.195397,-63.616672,null,100.992541,-95.712891,null,-3.74922,1.888334,null,108.277199,138.252924,null,-95.712891,133.775136,null,16.354896,8.468946,null,15.472962,10.451526,null,35.243322,21.824312,null,35.243322,28.369885,null,35.243322,-3.74922,null,10.451526,4.351721,null,1.888334,9.537499,null,43.356892,19.145136,null,78.96288,80.771797,null,19.145136,15.472962,null,104.195397,30.802498,null,27.953389,105.318756,null,46.869107,1.888334,null,24.96676,19.145136,null,-3.74922,12.56738,null,1.888334,-3.74922,null,104.195397,31.16558,null,-8.224454,1.888334,null,104.195397,-88.89653,null,78.96288,8.675277,null,104.195397,66.923684,null,104.195397,-8.24389,null,108.277199,10.451526,null,-3.435973,-3.74922,null,127.766922,-95.712891,null,104.195397,3.0,null,104.195397,-83.753428,null,15.472962,-95.712891,null,5.291266,12.56738,null,104.195397,24.96676,null,108.277199,127.766922,null,1.888334,4.351721,null,104.195397,-8.224454,null,1.888334,12.56738,null,104.195397,8.675277,null,19.145136,5.291266,null,12.56738,1.888334,null,5.291266,-95.712891,null,104.195397,15.472962,null,114.109497,104.195397,null,104.195397,-1.023194,null,127.766922,138.252924,null,-58.443832,-51.92528,null,10.451526,16.354896,null,104.195397,-70.162651,null,104.195397,14.550072,null,1.888334,5.291266,null,16.354896,9.501785,null,-3.435973,-8.24389,null,-3.435973,-8.224454,null,127.766922,114.109497,null,104.195397,-55.765835,null,127.766922,-1.023194,null,-95.712891,4.351721,null,16.354896,19.145136,null,35.243322,-3.435973,null,10.451526,-3.435973,null,104.195397,19.503304,null,35.243322,-8.224454,null,78.96288,10.451526,null,25.48583,5.291266,null,104.195397,17.228331,null,24.96676,1.888334,null,104.195397,90.3563,null,-95.712891,90.3563,null,15.472962,19.145136,null,19.145136,19.503304,null,105.318756,27.953389,null,-3.435973,10.451526,null,104.195397,23.881275,null,101.975766,35.243322,null,-3.74922,-7.09262,null,114.109497,-95.712891,null,78.96288,34.888822,null,78.96288,5.291266,null,24.96676,10.451526,null,9.537499,12.56738,null,78.96288,84.124008,null,108.277199,5.291266,null,103.819836,174.885971,null,19.145136,1.888334,null,28.369885,24.96676,null,4.351721,1.888334,null,104.195397,51.183884,null,10.451526,9.501785,null,127.766922,108.277199,null,100.992541,10.451526,null,-95.712891,-3.435973,null,35.243322,24.96676,null,16.354896,25.748151,null,5.291266,-3.74922,null,104.195397,53.688046,null,35.243322,25.48583,null,5.291266,-3.435973,null,104.195397,-86.241905,null,21.824312,19.503304,null,35.243322,17.228331,null,-95.712891,10.451526,null,-3.435973,5.291266,null,105.318756,66.923684,null,100.992541,15.472962,null,95.955974,127.766922,null,-95.712891,80.771797,null,4.351721,10.451526,null,78.96288,-3.435973,null,15.472962,19.699024,null,104.195397,47.481766,null,9.501785,8.468946,null,9.537499,-3.435973,null,19.145136,31.16558,null,35.243322,25.748151,null,19.145136,12.56738,null,12.56738,10.451526,null,104.195397,25.48583,null,104.195397,35.243322,null,104.195397,57.552152,null,34.851612,-78.183406,null,127.766922,10.451526,null,69.345116,-95.712891,null,-95.712891,-11.779889,null,104.195397,35.862285,null,5.291266,-8.24389,null,104.195397,69.345116,null,103.819836,101.975766,null,10.451526,-95.712891,null,15.472962,1.888334,null,35.243322,14.550072,null,27.953389,5.291266,null,104.195397,19.699024,null,24.603189,16.354896,null,23.881275,10.451526,null,35.243322,16.354896,null,138.252924,108.277199,null,4.351721,15.472962,null,100.992541,5.291266,null,10.451526,19.503304,null,104.195397,32.290275,null,-3.74922,10.451526,null,104.195397,34.888822,null,104.195397,36.238414,null,5.291266,35.243322,null,104.195397,-58.93018,null,104.195397,37.906193,null,104.990963,-106.346771,null,104.195397,-66.58973,null,1.888334,24.96676,null,19.503304,10.451526,null,1.888334,-8.224454,null,35.243322,19.503304,null,24.96676,12.56738,null,108.277199,-106.346771,null,21.005859,10.451526,null,104.195397,15.2,null,19.145136,16.354896,null,78.96288,1.888334,null,103.819836,-63.616672,null,138.252924,-95.712891,null,5.291266,15.472962,null,12.56738,19.503304,null,78.96288,138.252924,null,104.195397,103.846656,null,5.291266,23.881275,null,35.243322,43.679291,null,-95.712891,-88.49765,null,10.451526,14.995463,null,104.195397,55.923255,null,104.195397,46.869107,null,1.888334,-8.24389,null,127.766922,103.819836,null,1.888334,8.227512,null,31.16558,19.699024,null,19.145136,24.96676,null,43.356892,5.291266,null,104.990963,138.252924,null,78.96288,12.56738,null,9.537499,-95.712891,null,35.243322,4.351721,null,10.451526,-8.24389,null,35.243322,9.501785,null,-95.712891,103.819836,null,10.451526,24.96676,null,108.277199,133.775136,null,138.252924,133.775136,null,101.975766,114.109497,null,10.451526,25.748151,null,35.243322,47.481766,null,19.145136,19.699024,null,103.819836,100.992541,null,15.472962,-8.224454,null,10.451526,19.699024,null,19.503304,23.881275,null,127.766922,104.195397,null,104.195397,-63.588653,null,23.881275,16.354896,null,100.992541,138.252924,null,78.96288,0.824782,null,25.48583,10.451526,null,78.96288,37.906193,null,1.888334,-3.435973,null,19.145136,23.881275,null,104.195397,17.873887,null,21.005859,24.96676,null,4.351721,-3.74922,null,-95.712891,104.195397,null,35.243322,15.472962,null,31.16558,24.96676,null,10.451526,-102.552784,null,21.824312,12.56738,null,19.145136,14.550072,null,10.451526,6.129583,null,78.96288,-106.346771,null,108.277199,104.195397,null,104.195397,43.356892,null,108.277199,16.354896,null,127.766922,113.921327,null,-78.183406,-74.297333,null,19.145136,25.748151,null,12.56738,8.468946,null,-1.023194,-3.435973,null,-1.023194,10.451526,null,16.354896,5.291266,null,-102.552784,103.819836,null,43.356892,4.351721,null,5.291266,8.227512,null,19.699024,19.145136,null,4.351721,19.145136,null,-55.765835,-51.92528,null,-63.588653,-66.58973,null,138.252924,104.195397,null,-3.435973,-95.712891,null,114.109497,121.774017,null,35.243322,34.851612,null,1.888334,16.354896,null,19.503304,5.291266,null,101.975766,100.992541,null,-95.712891,138.252924,null,19.503304,19.699024,null,104.195397,127.510093,null,108.277199,-3.74922,null,19.503304,12.56738,null,31.16558,10.451526,null,21.824312,-3.74922,null,-7.09262,-8.224454,null,108.277199,114.109497,null,25.48583,6.129583,null,104.195397,14.375416,null,25.48583,21.824312,null,30.802498,114.109497,null,133.775136,174.885971,null,127.766922,101.975766,null,5.291266,9.501785,null,10.451526,105.318756,null,43.356892,10.451526,null,12.56738,8.227512,null,19.503304,24.96676,null,19.145136,-3.74922,null,19.699024,14.550072,null,35.243322,19.145136,null,-3.435973,4.351721,null,10.451526,-8.224454,null,104.195397,-7.09262,null,19.145136,4.351721,null,1.888334,19.145136,null,103.819836,-78.183406,null,10.451526,25.48583,null,10.451526,133.775136,null,127.766922,-9.696645,null,104.195397,50.55096,null,78.96288,133.775136,null,10.451526,23.881275,null,30.802498,35.243322,null,9.501785,16.354896,null,35.243322,23.881275,null,78.96288,100.992541,null,25.48583,12.56738,null,108.277199,19.145136,null,104.195397,80.771797,null,-8.224454,19.699024,null,101.975766,103.819836,null,10.451526,104.195397,null,104.195397,-14.452362,null,104.195397,47.576927,null,10.451526,22.937506,null,-3.74922,-3.435973,null,108.277199,113.921327,null,103.819836,-95.712891,null,104.195397,8.081666,null,104.195397,64.585262,null,35.243322,6.129583,null,104.195397,74.766098,null,104.195397,25.013607,null,127.766922,-102.552784,null,-3.435973,-7.09262,null,-95.712891,-83.753428,null,108.277199,1.888334,null,35.243322,43.356892,null,-51.92528,-63.616672,null,15.472962,4.351721,null,121.774017,-95.712891,null,35.243322,8.227512,null,12.56738,-3.435973,null,-95.712891,45.079162,null,1.888334,174.885971,null,1.888334,-95.712891,null,103.819836,-102.552784,null,103.819836,102.495496,null,35.243322,19.699024,null,-8.24389,138.252924,null,-8.224454,-3.435973,null,23.881275,25.013607,null,19.145136,-3.435973,null,5.291266,-102.552784,null,19.145136,8.468946,null,-3.435973,14.550072,null,35.243322,14.995463,null,104.195397,35.529562,null,10.451526,8.468946,null,104.195397,0.824782,null,28.369885,19.699024,null,113.921327,121.774017,null,25.013607,25.748151,null,100.992541,133.775136,null,19.145136,105.318756,null,104.195397,29.154857,null,104.195397,29.873888,null,127.766922,133.775136,null,15.2,10.451526,null,23.881275,19.145136,null,104.195397,12.354722,null,15.472962,12.56738,null,10.451526,15.2,null,103.819836,114.109497,null,25.48583,1.888334,null,19.145136,24.603189,null,16.354896,10.451526,null,66.923684,78.96288,null,69.345116,10.451526,null,22.937506,18.49041,null,12.56738,21.824312,null,-3.435973,19.145136,null,30.802498,8.675277,null,-95.712891,5.291266,null,15.472962,14.550072,null,8.468946,16.354896,null,-3.435973,53.847818,null,-90.230759,-86.241905,null,8.468946,12.56738,null,35.243322,21.005859,null,12.56738,19.145136,null,103.819836,113.921327,null,19.503304,15.2,null,19.503304,21.824312,null,9.537499,4.351721,null,-95.712891,-71.542969,null,133.775136,-95.712891,null,19.699024,1.888334,null,19.699024,19.503304,null,16.354896,-3.435973,null,12.56738,-3.74922,null,127.766922,-80.782127,null,104.990963,-3.435973,null,101.975766,133.775136,null,104.195397,48.516388,null,108.277199,105.318756,null,-102.552784,-90.230759,null,104.195397,178.065033,null,104.195397,42.590275,null,10.451526,21.824312,null,104.195397,24.603189,null,78.96288,95.955974,null,21.824312,19.699024,null,108.277199,12.56738,null,10.451526,53.847818,null,113.921327,103.819836,null,127.766922,100.992541,null,-51.92528,-55.765835,null,10.451526,35.243322,null,108.277199,-5.54708,null,31.16558,19.145136,null,64.585262,71.276093,null,34.851612,-95.712891,null,-3.435973,133.775136,null,114.109497,108.277199,null,5.291266,8.468946,null,104.195397,33.429859,null,-70.162651,12.56738,null,104.990963,104.195397,null,19.145136,-95.712891,null,23.881275,9.501785,null,-106.346771,-72.285215,null,5.291266,16.354896,null,105.318756,31.16558,null,104.195397,27.849332,null,8.227512,10.451526,null,-8.224454,10.451526,null,35.243322,69.345116,null,45.079162,35.862285,null,35.243322,47.576927,null,43.356892,-3.435973,null,24.96676,-3.74922,null,127.766922,121.774017,null,104.195397,143.95555,null,104.195397,21.005859,null,5.291266,24.96676,null,104.195397,-85.207229,null,78.96288,9.501785,null,10.451526,21.005859,null,10.451526,30.802498,null,104.195397,9.537499,null,35.243322,-8.24389,null,104.195397,84.124008,null,4.351721,12.56738,null,19.699024,-95.712891,null,10.451526,-7.09262,null,43.356892,1.888334,null,108.277199,104.990963,null,15.472962,-3.435973,null,9.501785,10.451526,null,24.603189,25.013607,null,104.195397,102.495496,null,104.195397,27.953389,null,-95.712891,22.937506,null,5.291266,-51.92528,null,104.195397,-61.222503,null,114.109497,-3.435973,null,108.277199,22.937506,null,35.243322,36.238414,null,108.277199,53.847818,null,23.881275,24.603189,null,37.906193,-66.58973,null,-61.222503,-58.93018,null,35.243322,3.0,null,4.351721,14.995463,null,-95.712891,101.975766,null,25.748151,10.451526,null,19.503304,1.888334,null,22.937506,29.154857,null,78.96288,2.315834,null,-7.09262,-3.435973,null,104.195397,24.15536,null,104.195397,46.199616,null,108.277199,90.3563,null,35.243322,-7.09262,null,24.96676,25.48583,null,78.96288,101.975766,null,15.2,14.995463,null,114.109497,113.543873,null,4.351721,6.129583,null,-51.92528,-95.712891,null,5.291266,21.824312,null,103.819836,121.774017,null,12.56738,24.96676,null,-7.09262,10.451526,null,104.195397,165.618042,null,35.243322,33.429859,null,25.748151,16.354896,null,69.345116,4.351721,null,69.345116,5.291266,null,9.501785,1.888334,null,-8.224454,4.351721,null,19.145136,14.995463,null,103.819836,138.252924,null,35.243322,133.775136,null,104.195397,-77.297508,null,9.537499,19.699024,null,104.195397,40.489673,null,78.96288,22.937506,null,35.243322,17.679076,null,5.291266,19.503304,null,104.195397,114.727669,null,138.252924,127.766922,null,17.679076,10.451526,null,113.921327,138.252924,null,114.109497,10.451526,null,5.291266,14.550072,null,-95.712891,-61.222503,null,104.195397,-9.696645,null,4.351721,-95.712891,null,19.145136,21.824312,null,69.345116,-3.435973,null,14.995463,15.2,null,-3.74922,5.291266,null,24.96676,4.351721,null,10.451526,31.16558,null,25.48583,4.351721,null,24.96676,5.291266,null,12.56738,20.0,null,-3.74922,104.195397,null,-95.712891,53.847818,null,50.55096,-95.712891,null,22.937506,24.684866,null,78.96288,65.0,null,138.252924,114.109497,null,127.766922,-106.346771,null,108.277199,100.992541,null,35.243322,21.745275,null,78.96288,90.3563,null,108.277199,101.975766,null,31.16558,35.243322,null,35.243322,15.2,null,24.96676,8.227512,null,78.96288,-80.782127,null,24.96676,-3.435973,null,-85.207229,-70.162651,null,103.819836,-83.753428,null,5.291266,25.748151,null,114.109497,138.252924,null,9.537499,36.238414,null,-8.224454,19.145136,null,78.96288,-8.224454,null,-95.712891,100.992541,null,21.005859,5.291266,null,37.906193,29.873888,null,23.881275,105.318756,null,19.699024,10.451526,null,1.888334,127.766922,null,104.195397,-5.54708,null,-95.712891,-80.782127,null,9.501785,90.3563,null,25.013607,16.354896,null,104.990963,10.451526,null,24.96676,19.699024,null,5.291266,19.699024,null,-74.297333,104.195397,null,-8.24389,-3.435973,null,108.277199,78.96288,null,10.451526,45.079162,null,27.953389,-75.015152,null,127.766922,-88.49765,null,24.603189,27.953389,null,-95.712891,-70.162651,null,25.48583,-102.552784,null,15.472962,-106.346771,null,78.96288,69.345116,null,108.277199,24.96676,null,127.766922,24.96676,null,9.537499,-3.74922,null,100.992541,53.688046,null,78.96288,-14.452362,null,5.291266,21.005859,null,138.252924,113.921327,null,9.501785,105.318756,null,101.975766,78.96288,null,1.888334,23.881275,null,101.975766,138.252924,null,16.354896,-19.020835,null,19.145136,27.953389,null,-106.346771,4.351721,null,103.819836,127.766922,null,4.351721,9.537499,null,108.277199,4.351721,null,-88.89653,-85.207229,null,113.921327,55.491977,null,104.195397,59.556278,null,104.195397,73.22068,null,-3.435973,138.252924,null,-102.552784,-106.346771,null,35.243322,-3.996166,null,24.96676,21.824312,null,127.766922,78.96288,null,35.243322,-1.023194,null,1.888334,-149.406843,null,19.503304,-3.74922,null,5.291266,-106.346771,null,104.195397,113.543873,null,14.995463,10.451526,null,-95.712891,35.243322,null,24.603189,23.881275,null,-3.74922,16.354896,null,5.291266,-19.020835,null,28.369885,105.318756,null,10.451526,9.537499,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,46.603354,40.463667,-11.202692,35.86166,39.399872,25.354826,-30.559482,37.09024,6.42375,-38.416097,-35.675147,-23.442503,-9.189967,-32.522779,-25.274398,50.850346,-9.64571,56.130366,-21.236736,61.92411,51.165691,7.946527,22.396428,-0.789275,7.539989,36.204824,35.907757,29.31166,-18.766947,4.210484,-22.95764,52.132633,-20.904305,-40.900557,8.537981,-6.314993,12.879721,1.352083,15.870032,23.424076,1.373333,26.820553,55.378051,26.02751,39.074208,21.512583,40.069099,53.709807,61.52401,13.193887,63.397768,33.0,28.0,47.516231,-14.235004,42.733883,7.369722,19.3133,6.611111,7.873054,4.570868,-0.228021,-2.1646,45.1,35.126413,49.817492,9.30769,56.26392,18.735693,-1.831239,9.145,58.595272,-0.803689,36.140751,15.783471,9.945587,18.971187,47.162494,64.963051,53.41291,31.046051,41.87194,48.019573,33.854721,56.879635,26.3351,55.169438,49.815273,17.570692,35.937496,23.634501,46.862496,47.411631,31.791702,9.081999,60.472024,30.375321,51.919438,45.943161,23.885942,14.497401,44.016521,8.460555,20.593684,48.669026,46.151241,3.919305,46.818188,33.886917,38.963745,48.379433,41.608635,-16.290154,43.915886,42.708678,-22.328474,17.060816,25.03428,9.748917,13.794185,-17.713371,4.860416,15.199999,18.109581,6.428055,21.00789,12.865416,7.131474,12.984305,41.0,40.143105,42.315407,14.058324,21.916221,12.565679,21.521757,-20.348404,12.16957,12.52111,46.946947,23.685,32.3078,27.514162,17.189877,4.535277,-3.373056,16.5388,15.454166,15.414999,1.650801,15.179384,-17.679742,11.825138,13.443182,31.952162,32.427908,33.223191,30.585164,-0.023559,41.20438,19.85627,-29.609988,22.198745,-13.254308,3.202778,-18.665695,28.394857,-15.376706,17.607789,11.803749,-8.874217,-1.940278,13.909444,0.18636,-4.679574,5.152149,-19.015438,7.862684,12.862807,-26.522503,34.802075,38.861034,8.619543,-21.178986,10.691803,38.969719,21.694025,-6.369028,12.238333,41.377491,-13.768752,-13.759029,15.552727,-13.133897,61.892635,71.706936,12.262776,17.357822,-9.2,-11.6455,13.444304,16.742498,43.94236,-3.370417,18.420695,12.20189,-19.054445,17.664332,-49.280366,-15.965,18.04248,-51.796253,-10.447525,10.961632,-0.522778,7.425554,-29.040835,-7.109535,17.897476,41.902916,40.339852,7.51498,-14.28522,18.218785,-12.164165,-90.0],"lon":[1.601554,1.888334,-3.74922,17.873887,104.195397,-8.224454,51.183884,22.937506,-95.712891,-66.58973,-63.616672,-71.542969,-58.443832,-75.015152,-55.765835,133.775136,4.351721,160.156194,-106.346771,-159.777671,25.748151,10.451526,-1.023194,114.109497,113.921327,-5.54708,138.252924,127.766922,47.481766,46.869107,101.975766,18.49041,5.291266,165.618042,174.885971,-80.782127,143.95555,121.774017,103.819836,100.992541,53.847818,32.290275,30.802498,-3.435973,50.55096,21.824312,55.923255,45.038189,27.953389,105.318756,-59.543198,16.354896,65.0,3.0,14.550072,-51.92528,25.48583,12.354722,-81.2546,20.939444,80.771797,-74.297333,15.827659,24.15536,15.2,33.429859,15.472962,2.315834,9.501785,-70.162651,-78.183406,40.489673,25.013607,11.609444,-5.353585,-90.230759,-9.696645,-72.285215,19.503304,-19.020835,-8.24389,34.851612,12.56738,66.923684,35.862285,24.603189,17.228331,23.881275,6.129583,-3.996166,14.375416,-102.552784,103.846656,28.369885,-7.09262,8.675277,8.468946,69.345116,19.145136,24.96676,45.079162,-14.452362,21.005859,-11.779889,78.96288,19.699024,14.995463,-56.027783,8.227512,9.537499,35.243322,31.16558,21.745275,-63.588653,17.679076,19.37439,24.684866,-61.796428,-77.39628,-83.753428,-88.89653,178.065033,-58.93018,-86.241905,-77.297508,-9.429499,-10.940835,-85.207229,171.184478,-61.287228,20.0,47.576927,43.356892,108.277199,95.955974,104.990963,-77.781167,57.552152,-68.990021,-69.968338,-56.32509,90.3563,-64.7505,90.433601,-88.49765,114.727669,29.918886,-23.0418,18.732207,-61.370976,10.267895,39.782334,-149.406843,42.590275,-15.310139,35.233154,53.688046,43.679291,36.238414,37.906193,74.766098,102.495496,28.233608,113.543873,34.301525,73.22068,35.529562,84.124008,166.959158,8.081666,-15.180413,125.727,29.873888,-60.978893,6.613081,55.491977,46.199616,29.154857,30.217636,30.217636,31.465866,38.996815,71.276093,0.824782,-175.198242,-61.222503,59.556278,-71.797928,34.888822,-1.561593,64.585262,-177.156097,-172.104629,48.516388,27.849332,-6.911806,-42.604303,-61.604171,-62.782998,-171.833333,43.3333,144.793731,-62.187366,12.457777,-168.734039,-64.639968,-68.262383,-169.867233,145.94351,69.348557,-5.7089,-63.05483,-59.523613,105.690449,-169.09022,166.931503,150.550812,167.954712,179.194167,-62.83055,12.453389,127.510093,134.58252,-170.70444,-63.043653,96.870956,0.0],"marker":{"color":"red","opacity":0.8,"size":[8.4,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,11.6,19.2,20.0,15.6,19.6,20.0,20.0,20.0,3.6,20.0,2.0,20.0,20.0,20.0,20.0,20.0,13.6,20.0,20.0,20.0,16.4,20.0,20.0,20.0,10.4,20.0,20.0,9.2,20.0,20.0,20.0,20.0,10.8,20.0,20.0,20.0,20.0,16.4,14.0,20.0,20.0,15.6,20.0,8.4,12.4,20.0,20.0,20.0,10.0,6.8,5.6,13.6,20.0,11.6,14.8,20.0,20.0,20.0,8.4,20.0,20.0,20.0,10.0,20.0,7.6,6.8,20.0,8.8,7.2,20.0,17.2,20.0,20.0,20.0,20.0,16.0,20.0,12.4,20.0,20.0,10.0,20.0,20.0,16.8,20.0,20.0,16.8,20.0,20.0,20.0,20.0,20.0,12.8,20.0,6.8,20.0,20.0,20.0,6.8,20.0,20.0,20.0,20.0,20.0,16.0,20.0,16.0,14.0,8.0,7.6,14.0,18.0,14.8,11.6,9.6,10.0,7.6,8.4,15.2,3.2,6.8,14.0,18.0,20.0,20.0,10.8,20.0,10.8,20.0,7.6,7.2,1.2,10.8,7.2,3.6,5.6,11.6,7.2,5.2,5.2,4.8,7.6,3.6,8.8,7.6,5.6,4.8,6.0,13.2,15.2,20.0,16.0,5.6,5.2,10.4,8.4,10.0,16.8,9.2,4.0,8.0,3.2,4.4,10.0,5.2,3.2,10.4,7.6,10.0,3.6,9.6,4.8,2.8,7.2,7.6,1.2,19.6,6.0,2.8,16.0,10.4,16.8,1.2,3.6,7.2,14.0,5.6,6.8,6.0,3.2,0.4,3.2,4.4,1.6,4.8,2.4,4.4,2.8,0.8,2.4,1.2,1.2,4.4,0.4,0.4,0.4,0.8,2.8,1.2,1.2,1.6,0.4,0.4,2.0,0.4,1.6,0.4,0.8]},"mode":"markers","name":"2021","text":["AND\u003cbr\u003eConex\u00f5es: 21","FRA\u003cbr\u003eConex\u00f5es: 251","ESP\u003cbr\u003eConex\u00f5es: 227","AGO\u003cbr\u003eConex\u00f5es: 55","CHN\u003cbr\u003eConex\u00f5es: 274","PRT\u003cbr\u003eConex\u00f5es: 192","QAT\u003cbr\u003eConex\u00f5es: 51","ZAF\u003cbr\u003eConex\u00f5es: 158","USA\u003cbr\u003eConex\u00f5es: 266","VEN\u003cbr\u003eConex\u00f5es: 29","ARG\u003cbr\u003eConex\u00f5es: 48","CHL\u003cbr\u003eConex\u00f5es: 79","PRY\u003cbr\u003eConex\u00f5es: 39","PER\u003cbr\u003eConex\u00f5es: 49","URY\u003cbr\u003eConex\u00f5es: 53","AUS\u003cbr\u003eConex\u00f5es: 148","BEL\u003cbr\u003eConex\u00f5es: 215","SLB\u003cbr\u003eConex\u00f5es: 9","CAN\u003cbr\u003eConex\u00f5es: 196","COK\u003cbr\u003eConex\u00f5es: 5","FIN\u003cbr\u003eConex\u00f5es: 147","DEU\u003cbr\u003eConex\u00f5es: 265","GHA\u003cbr\u003eConex\u00f5es: 56","HKG\u003cbr\u003eConex\u00f5es: 192","IDN\u003cbr\u003eConex\u00f5es: 140","CIV\u003cbr\u003eConex\u00f5es: 34","JPN\u003cbr\u003eConex\u00f5es: 156","KOR\u003cbr\u003eConex\u00f5es: 221","KWT\u003cbr\u003eConex\u00f5es: 55","MDG\u003cbr\u003eConex\u00f5es: 41","MYS\u003cbr\u003eConex\u00f5es: 94","NAM\u003cbr\u003eConex\u00f5es: 59","NLD\u003cbr\u003eConex\u00f5es: 238","NCL\u003cbr\u003eConex\u00f5es: 26","NZL\u003cbr\u003eConex\u00f5es: 114","PAN\u003cbr\u003eConex\u00f5es: 51","PNG\u003cbr\u003eConex\u00f5es: 23","PHL\u003cbr\u003eConex\u00f5es: 104","SGP\u003cbr\u003eConex\u00f5es: 161","THA\u003cbr\u003eConex\u00f5es: 160","ARE\u003cbr\u003eConex\u00f5es: 82","UGA\u003cbr\u003eConex\u00f5es: 27","EGY\u003cbr\u003eConex\u00f5es: 88","GBR\u003cbr\u003eConex\u00f5es: 232","BHR\u003cbr\u003eConex\u00f5es: 50","GRC\u003cbr\u003eConex\u00f5es: 133","OMN\u003cbr\u003eConex\u00f5es: 41","ARM\u003cbr\u003eConex\u00f5es: 35","BLR\u003cbr\u003eConex\u00f5es: 80","RUS\u003cbr\u003eConex\u00f5es: 158","BRB\u003cbr\u003eConex\u00f5es: 39","SWE\u003cbr\u003eConex\u00f5es: 185","AFG\u003cbr\u003eConex\u00f5es: 21","DZA\u003cbr\u003eConex\u00f5es: 31","AUT\u003cbr\u003eConex\u00f5es: 57","BRA\u003cbr\u003eConex\u00f5es: 149","BGR\u003cbr\u003eConex\u00f5es: 123","CMR\u003cbr\u003eConex\u00f5es: 25","CYM\u003cbr\u003eConex\u00f5es: 17","CAF\u003cbr\u003eConex\u00f5es: 14","LKA\u003cbr\u003eConex\u00f5es: 34","COL\u003cbr\u003eConex\u00f5es: 93","COG\u003cbr\u003eConex\u00f5es: 29","COD\u003cbr\u003eConex\u00f5es: 37","HRV\u003cbr\u003eConex\u00f5es: 100","CYP\u003cbr\u003eConex\u00f5es: 63","CZE\u003cbr\u003eConex\u00f5es: 181","BEN\u003cbr\u003eConex\u00f5es: 21","DNK\u003cbr\u003eConex\u00f5es: 208","DOM\u003cbr\u003eConex\u00f5es: 65","ECU\u003cbr\u003eConex\u00f5es: 61","ETH\u003cbr\u003eConex\u00f5es: 25","EST\u003cbr\u003eConex\u00f5es: 98","GAB\u003cbr\u003eConex\u00f5es: 19","GIB\u003cbr\u003eConex\u00f5es: 17","GTM\u003cbr\u003eConex\u00f5es: 55","GIN\u003cbr\u003eConex\u00f5es: 22","HTI\u003cbr\u003eConex\u00f5es: 18","HUN\u003cbr\u003eConex\u00f5es: 142","ISL\u003cbr\u003eConex\u00f5es: 43","IRL\u003cbr\u003eConex\u00f5es: 132","ISR\u003cbr\u003eConex\u00f5es: 112","ITA\u003cbr\u003eConex\u00f5es: 215","KAZ\u003cbr\u003eConex\u00f5es: 78","LBN\u003cbr\u003eConex\u00f5es: 40","LVA\u003cbr\u003eConex\u00f5es: 103","LBY\u003cbr\u003eConex\u00f5es: 31","LTU\u003cbr\u003eConex\u00f5es: 126","LUX\u003cbr\u003eConex\u00f5es: 109","MLI\u003cbr\u003eConex\u00f5es: 25","MLT\u003cbr\u003eConex\u00f5es: 57","MEX\u003cbr\u003eConex\u00f5es: 87","MNG\u003cbr\u003eConex\u00f5es: 42","MDA\u003cbr\u003eConex\u00f5es: 54","MAR\u003cbr\u003eConex\u00f5es: 92","NGA\u003cbr\u003eConex\u00f5es: 42","NOR\u003cbr\u003eConex\u00f5es: 124","PAK\u003cbr\u003eConex\u00f5es: 142","POL\u003cbr\u003eConex\u00f5es: 198","ROU\u003cbr\u003eConex\u00f5es: 129","SAU\u003cbr\u003eConex\u00f5es: 60","SEN\u003cbr\u003eConex\u00f5es: 32","SRB\u003cbr\u003eConex\u00f5es: 101","SLE\u003cbr\u003eConex\u00f5es: 17","IND\u003cbr\u003eConex\u00f5es: 230","SVK\u003cbr\u003eConex\u00f5es: 122","SVN\u003cbr\u003eConex\u00f5es: 128","SUR\u003cbr\u003eConex\u00f5es: 17","CHE\u003cbr\u003eConex\u00f5es: 214","TUN\u003cbr\u003eConex\u00f5es: 93","TUR\u003cbr\u003eConex\u00f5es: 235","UKR\u003cbr\u003eConex\u00f5es: 132","MKD\u003cbr\u003eConex\u00f5es: 60","BOL\u003cbr\u003eConex\u00f5es: 40","BIH\u003cbr\u003eConex\u00f5es: 62","MNE\u003cbr\u003eConex\u00f5es: 40","BWA\u003cbr\u003eConex\u00f5es: 35","ATG\u003cbr\u003eConex\u00f5es: 20","BHS\u003cbr\u003eConex\u00f5es: 19","CRI\u003cbr\u003eConex\u00f5es: 35","SLV\u003cbr\u003eConex\u00f5es: 45","FJI\u003cbr\u003eConex\u00f5es: 37","GUY\u003cbr\u003eConex\u00f5es: 29","HND\u003cbr\u003eConex\u00f5es: 24","JAM\u003cbr\u003eConex\u00f5es: 25","LBR\u003cbr\u003eConex\u00f5es: 19","MRT\u003cbr\u003eConex\u00f5es: 21","NIC\u003cbr\u003eConex\u00f5es: 38","MHL\u003cbr\u003eConex\u00f5es: 8","VCT\u003cbr\u003eConex\u00f5es: 17","ALB\u003cbr\u003eConex\u00f5es: 35","AZE\u003cbr\u003eConex\u00f5es: 45","GEO\u003cbr\u003eConex\u00f5es: 67","VNM\u003cbr\u003eConex\u00f5es: 130","MMR\u003cbr\u003eConex\u00f5es: 27","KHM\u003cbr\u003eConex\u00f5es: 57","CUB\u003cbr\u003eConex\u00f5es: 27","MUS\u003cbr\u003eConex\u00f5es: 50","CUW\u003cbr\u003eConex\u00f5es: 19","ABW\u003cbr\u003eConex\u00f5es: 18","SPM\u003cbr\u003eConex\u00f5es: 3","BGD\u003cbr\u003eConex\u00f5es: 27","BMU\u003cbr\u003eConex\u00f5es: 18","BTN\u003cbr\u003eConex\u00f5es: 9","BLZ\u003cbr\u003eConex\u00f5es: 14","BRN\u003cbr\u003eConex\u00f5es: 29","BDI\u003cbr\u003eConex\u00f5es: 18","CPV\u003cbr\u003eConex\u00f5es: 13","TCD\u003cbr\u003eConex\u00f5es: 13","DMA\u003cbr\u003eConex\u00f5es: 12","GNQ\u003cbr\u003eConex\u00f5es: 19","ERI\u003cbr\u003eConex\u00f5es: 9","PYF\u003cbr\u003eConex\u00f5es: 22","DJI\u003cbr\u003eConex\u00f5es: 19","GMB\u003cbr\u003eConex\u00f5es: 14","PSE\u003cbr\u003eConex\u00f5es: 12","IRN\u003cbr\u003eConex\u00f5es: 15","IRQ\u003cbr\u003eConex\u00f5es: 33","JOR\u003cbr\u003eConex\u00f5es: 38","KEN\u003cbr\u003eConex\u00f5es: 88","KGZ\u003cbr\u003eConex\u00f5es: 40","LAO\u003cbr\u003eConex\u00f5es: 14","LSO\u003cbr\u003eConex\u00f5es: 13","MAC\u003cbr\u003eConex\u00f5es: 26","MWI\u003cbr\u003eConex\u00f5es: 21","MDV\u003cbr\u003eConex\u00f5es: 25","MOZ\u003cbr\u003eConex\u00f5es: 42","NPL\u003cbr\u003eConex\u00f5es: 23","VUT\u003cbr\u003eConex\u00f5es: 10","NER\u003cbr\u003eConex\u00f5es: 20","GNB\u003cbr\u003eConex\u00f5es: 8","TLS\u003cbr\u003eConex\u00f5es: 11","RWA\u003cbr\u003eConex\u00f5es: 25","LCA\u003cbr\u003eConex\u00f5es: 13","STP\u003cbr\u003eConex\u00f5es: 8","SYC\u003cbr\u003eConex\u00f5es: 26","SOM\u003cbr\u003eConex\u00f5es: 19","ZWE\u003cbr\u003eConex\u00f5es: 25","SSD\u003cbr\u003eConex\u00f5es: 9","SDN\u003cbr\u003eConex\u00f5es: 24","SWZ\u003cbr\u003eConex\u00f5es: 12","SYR\u003cbr\u003eConex\u00f5es: 7","TJK\u003cbr\u003eConex\u00f5es: 18","TGO\u003cbr\u003eConex\u00f5es: 19","TON\u003cbr\u003eConex\u00f5es: 3","TTO\u003cbr\u003eConex\u00f5es: 49","TKM\u003cbr\u003eConex\u00f5es: 15","TCA\u003cbr\u003eConex\u00f5es: 7","TZA\u003cbr\u003eConex\u00f5es: 40","BFA\u003cbr\u003eConex\u00f5es: 26","UZB\u003cbr\u003eConex\u00f5es: 42","WLF\u003cbr\u003eConex\u00f5es: 3","WSM\u003cbr\u003eConex\u00f5es: 9","YEM\u003cbr\u003eConex\u00f5es: 18","ZMB\u003cbr\u003eConex\u00f5es: 35","FRO\u003cbr\u003eConex\u00f5es: 14","GRL\u003cbr\u003eConex\u00f5es: 17","GRD\u003cbr\u003eConex\u00f5es: 15","KNA\u003cbr\u003eConex\u00f5es: 8","TKL\u003cbr\u003eConex\u00f5es: 1","COM\u003cbr\u003eConex\u00f5es: 8","GUM\u003cbr\u003eConex\u00f5es: 11","MSR\u003cbr\u003eConex\u00f5es: 4","SMR\u003cbr\u003eConex\u00f5es: 12","KIR\u003cbr\u003eConex\u00f5es: 6","VGB\u003cbr\u003eConex\u00f5es: 11","BES\u003cbr\u003eConex\u00f5es: 7","NIU\u003cbr\u003eConex\u00f5es: 2","MNP\u003cbr\u003eConex\u00f5es: 6","ATF\u003cbr\u003eConex\u00f5es: 3","SHN\u003cbr\u003eConex\u00f5es: 3","SXM\u003cbr\u003eConex\u00f5es: 11","FLK\u003cbr\u003eConex\u00f5es: 1","CXR\u003cbr\u003eConex\u00f5es: 1","UMI\u003cbr\u003eConex\u00f5es: 1","NRU\u003cbr\u003eConex\u00f5es: 2","FSM\u003cbr\u003eConex\u00f5es: 7","NFK\u003cbr\u003eConex\u00f5es: 3","TUV\u003cbr\u003eConex\u00f5es: 3","BLM\u003cbr\u003eConex\u00f5es: 4","VAT\u003cbr\u003eConex\u00f5es: 1","PRK\u003cbr\u003eConex\u00f5es: 1","PLW\u003cbr\u003eConex\u00f5es: 5","ASM\u003cbr\u003eConex\u00f5es: 1","AIA\u003cbr\u003eConex\u00f5es: 4","CCK\u003cbr\u003eConex\u00f5es: 1","ATA\u003cbr\u003eConex\u00f5es: 2"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null,35.86166,36.204824,null,50.850346,56.26392,null,23.634501,37.09024,null,37.09024,23.634501,null,35.86166,51.165691,null,35.86166,55.378051,null,35.86166,15.870032,null,52.132633,51.165691,null,35.86166,-25.274398,null,35.86166,52.132633,null,35.86166,56.130366,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.5523254698666666},"lon":[104.195397,-95.712891,null,104.195397,138.252924,null,4.351721,9.501785,null,-102.552784,-95.712891,null,-95.712891,-102.552784,null,104.195397,10.451526,null,104.195397,-3.435973,null,104.195397,100.992541,null,5.291266,10.451526,null,104.195397,133.775136,null,104.195397,5.291266,null,104.195397,-106.346771,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,35.907757,null,52.132633,50.850346,null,37.09024,56.130366,null,35.86166,46.603354,null,35.86166,12.879721,null,35.86166,-0.789275,null,35.86166,23.634501,null,35.86166,40.463667,null,35.86166,51.919438,null,14.058324,37.09024,null,35.86166,20.593684,null,35.907757,35.86166,null,35.86166,61.52401,null,35.86166,50.850346,null,51.165691,46.603354,null,51.165691,51.919438,null,35.86166,4.210484,null,52.132633,46.603354,null,51.919438,51.165691,null,35.86166,63.397768,null,35.86166,23.885942,null,35.86166,41.87194,null,35.86166,14.058324,null,30.375321,37.09024,null,35.86166,-35.675147,null,18.735693,37.09024,null,56.130366,37.09024,null,20.593684,37.09024,null,55.169438,63.397768,null,12.879721,36.204824,null,35.86166,-14.235004,null,33.886917,46.603354,null,51.165691,47.516231,null,46.603354,40.463667,null,15.870032,37.09024,null,35.86166,1.352083,null,51.165691,41.87194,null,38.963745,37.09024,null,14.058324,36.204824,null,51.165691,40.463667,null,35.86166,23.424076,null,12.565679,37.09024,null,38.963745,51.165691,null,51.165691,46.818188,null,35.86166,48.019573,null,35.86166,-30.559482,null,51.165691,52.132633,null,40.463667,46.603354,null,51.165691,49.817492,null,35.86166,22.396428,null,51.919438,49.817492,null,14.058324,51.165691,null,35.86166,26.820553,null,40.463667,39.399872,null,14.058324,35.907757,null,31.791702,40.463667,null,53.709807,61.52401,null,35.86166,39.074208,null,52.132633,51.919438,null,52.132633,41.87194,null,40.463667,41.87194,null,40.463667,31.791702,null,35.86166,-1.831239,null,35.86166,-9.189967,null,51.165691,35.86166,null,35.86166,33.223191,null,42.315407,51.919438,null,-18.766947,46.603354,null,63.397768,60.472024,null,20.593684,9.081999,null,39.074208,47.162494,null,49.817492,51.165691,null,52.132633,35.86166,null,51.165691,63.397768,null,30.375321,50.850346,null,51.165691,50.850346,null,52.132633,40.463667,null,35.86166,8.537981,null,37.09024,35.86166,null,35.86166,4.570868,null,35.86166,31.046051,null,35.86166,46.151241,null,35.86166,56.26392,null,39.074208,41.87194,null,21.916221,37.09024,null,35.907757,22.396428,null,45.943161,51.165691,null,47.411631,45.943161,null,38.963745,47.411631,null,4.210484,38.963745,null,37.09024,8.619543,null,-23.442503,-14.235004,null,51.165691,22.396428,null,46.603354,41.87194,null,35.86166,-40.900557,null,36.204824,14.058324,null,50.850346,51.165691,null,35.86166,9.081999,null,50.850346,46.603354,null,35.86166,46.818188,null,35.86166,60.472024,null,37.09024,50.850346,null,20.593684,23.424076,null,52.132633,37.09024,null,51.919438,48.379433,null,35.86166,47.516231,null,35.86166,21.916221,null,45.943161,51.919438,null,4.210484,46.603354,null,22.396428,35.86166,null,50.850346,52.132633,null,35.907757,36.204824,null,47.516231,51.165691,null,35.86166,53.41291,null,39.399872,46.603354,null,51.165691,55.378051,null,38.963745,41.87194,null,35.86166,23.685,null,51.919438,52.132633,null,20.593684,52.132633,null,46.603354,51.165691,null,49.817492,51.919438,null,63.397768,51.919438,null,51.165691,56.26392,null,35.86166,39.399872,null,38.963745,52.132633,null,38.963745,39.074208,null,14.058324,35.86166,null,39.399872,49.817492,null,33.886917,41.87194,null,63.397768,61.92411,null,35.907757,7.946527,null,35.86166,29.31166,null,38.963745,40.463667,null,51.165691,47.162494,null,12.565679,35.86166,null,35.86166,13.794185,null,51.919438,48.669026,null,37.09024,-35.675147,null,35.86166,55.169438,null,20.593684,-6.369028,null,39.399872,40.463667,null,35.86166,12.565679,null,52.132633,53.41291,null,51.919438,47.162494,null,14.058324,55.378051,null,35.86166,-38.416097,null,21.916221,35.907757,null,35.86166,47.162494,null,31.791702,55.378051,null,41.377491,38.861034,null,35.907757,14.058324,null,38.963745,46.603354,null,37.09024,8.537981,null,31.046051,-1.831239,null,63.397768,56.26392,null,45.943161,46.603354,null,35.86166,42.733883,null,56.26392,46.603354,null,46.603354,50.850346,null,41.87194,46.603354,null,14.058324,52.132633,null,49.817492,37.09024,null,47.162494,55.169438,null,20.593684,-0.023559,null,45.943161,41.87194,null,52.132633,55.378051,null,52.132633,56.26392,null,51.919438,46.603354,null,12.565679,36.204824,null,46.603354,33.886917,null,14.058324,40.463667,null,35.86166,-6.369028,null,52.132633,49.817492,null,51.165691,33.886917,null,35.86166,30.375321,null,15.870032,49.817492,null,37.09024,55.378051,null,50.850346,49.815273,null,38.963745,45.943161,null,35.907757,37.09024,null,50.850346,49.817492,null,52.132633,39.399872,null,30.375321,52.132633,null,48.669026,47.516231,null,52.132633,58.595272,null,46.603354,52.132633,null,35.86166,61.92411,null,38.963745,47.162494,null,51.165691,37.09024,null,22.396428,14.058324,null,51.919438,56.26392,null,51.919438,50.850346,null,36.204824,35.86166,null,51.919438,61.92411,null,51.919438,63.397768,null,55.378051,53.41291,null,49.817492,48.669026,null,35.86166,9.748917,null,56.130366,51.165691,null,37.09024,-25.274398,null,35.86166,45.943161,null,20.593684,41.87194,null,20.593684,51.165691,null,20.593684,15.870032,null,51.165691,45.943161,null,35.86166,-11.202692,null,42.733883,51.165691,null,42.315407,52.132633,null,22.396428,37.09024,null,30.375321,55.378051,null,14.058324,51.919438,null,35.86166,12.862807,null,20.593684,46.603354,null,14.058324,46.603354,null,51.165691,23.634501,null,51.919438,45.943161,null,52.132633,56.879635,null,51.919438,41.87194,null,55.378051,51.165691,null,39.399872,48.669026,null,1.352083,36.204824,null,52.132633,61.92411,null,35.86166,25.354826,null,38.963745,55.378051,null,35.86166,38.963745,null,48.379433,48.669026,null,63.397768,51.165691,null,35.86166,7.873054,null,51.919438,37.09024,null,46.603354,35.86166,null,52.132633,45.943161,null,41.87194,51.165691,null,46.603354,46.818188,null,51.165691,53.41291,null,30.375321,42.733883,null,46.603354,45.943161,null,52.132633,63.397768,null,-25.274398,-13.759029,null,38.963745,61.52401,null,52.132633,23.424076,null,20.593684,56.130366,null,7.946527,51.165691,null,51.165691,48.669026,null,14.058324,63.397768,null,35.86166,32.427908,null,4.210484,-25.274398,null,38.963745,23.885942,null,38.963745,42.733883,null,40.463667,55.378051,null,48.379433,51.165691,null,20.593684,55.378051,null,47.162494,45.943161,null,20.593684,-25.274398,null,38.963745,33.223191,null,56.26392,60.472024,null,40.463667,51.165691,null,33.886917,40.463667,null,35.86166,-16.290154,null,42.315407,51.165691,null,46.603354,22.396428,null,39.074208,40.463667,null,51.165691,46.151241,null,37.09024,51.165691,null,56.879635,63.397768,null,51.919438,47.516231,null,35.86166,28.0,null,51.165691,55.169438,null,47.411631,48.669026,null,46.603354,51.919438,null,51.165691,31.791702,null,39.399872,51.919438,null,23.634501,18.735693,null,35.86166,31.791702,null,51.165691,45.1,null,52.132633,38.963745,null,14.058324,12.565679,null,47.162494,51.165691,null,35.86166,26.3351,null,52.132633,55.169438,null,35.86166,15.783471,null,35.907757,-0.789275,null,38.963745,31.046051,null,38.963745,12.862807,null,1.352083,4.210484,null,15.870032,36.204824,null,30.375321,51.919438,null,37.09024,51.919438,null,35.86166,-32.522779,null,52.132633,47.162494,null,51.165691,61.92411,null,35.86166,48.669026,null,-0.789275,36.204824,null,30.375321,51.165691,null,22.396428,22.198745,null,35.86166,49.817492,null,48.669026,49.817492,null,4.210484,1.352083,null,56.26392,51.165691,null,52.132633,48.669026,null,33.886917,50.850346,null,35.86166,22.198745,null,1.352083,-25.274398,null,23.634501,9.748917,null,39.399872,51.165691,null,35.907757,12.565679,null,20.593684,4.210484,null,33.886917,48.669026,null,4.210484,22.396428,null,39.399872,55.378051,null,1.352083,22.396428,null,52.132633,47.516231,null,14.058324,56.130366,null,37.09024,36.204824,null,51.919438,55.169438,null,46.603354,55.378051,null,55.378051,46.603354,null,51.165691,39.399872,null,41.87194,35.86166,null,49.817492,56.26392,null,4.210484,35.86166,null,55.378051,52.132633,null,20.593684,30.375321,null,47.516231,47.162494,null,38.963745,49.817492,null,51.165691,-14.235004,null,35.86166,15.199999,null,52.132633,49.815273,null,1.352083,35.86166,null,14.058324,23.634501,null,56.130366,8.619543,null,38.963745,47.516231,null,56.26392,63.397768,null,42.315407,47.162494,null,63.397768,56.879635,null,61.92411,63.397768,null,35.86166,30.585164,null,35.86166,-2.1646,null,14.058324,41.87194,null,20.593684,36.204824,null,35.86166,45.1,null,31.046051,35.126413,null,-32.522779,-14.235004,null,35.907757,46.862496,null,38.963745,40.143105,null,38.963745,26.3351,null,-25.274398,-40.900557,null,51.919438,56.879635,null,53.41291,-0.023559,null,49.817492,46.603354,null,20.593684,23.685,null,35.86166,18.735693,null,48.379433,45.943161,null,42.315407,49.817492,null,30.375321,56.130366,null,35.86166,41.377491,null,52.132633,39.074208,null,14.058324,-0.789275,null,53.41291,-19.015438,null,35.86166,10.691803,null,35.86166,14.497401,null,40.463667,-35.675147,null,51.919438,40.463667,null,50.850346,40.463667,null,31.791702,39.399872,null,52.132633,46.818188,null,51.919438,55.378051,null,15.870032,51.165691,null,51.165691,38.963745,null,41.87194,46.818188,null,30.375321,46.603354,null,37.09024,48.019573,null,37.09024,52.132633,null,37.09024,22.396428,null,49.817492,40.463667,null,50.850346,51.919438,null,14.058324,50.850346,null,37.09024,18.735693,null,38.963745,31.791702,null,20.593684,23.885942,null,51.165691,49.815273,null,35.86166,-0.023559,null,12.879721,20.593684,null,45.943161,38.963745,null,38.963745,48.669026,null,35.86166,-18.766947,null,22.396428,51.165691,null,35.86166,48.379433,null,22.396428,36.204824,null,56.26392,53.41291,null,49.817492,47.516231,null,15.870032,-25.274398,null,48.669026,47.162494,null,51.919438,35.86166,null,52.132633,56.130366,null,50.850346,33.886917,null,48.669026,51.165691,null,39.074208,48.379433,null,31.046051,-32.522779,null,51.165691,26.820553,null,41.87194,40.463667,null,42.315407,50.850346,null,46.603354,47.516231,null,35.86166,53.709807,null,55.169438,56.26392,null,40.463667,35.86166,null,55.378051,22.396428,null,20.593684,21.512583,null,45.943161,50.850346,null,38.963745,42.315407,null,35.86166,6.42375,null,55.378051,-30.559482,null,46.603354,-40.900557,null,49.817492,50.850346,null,48.669026,51.919438,null,-30.559482,-22.95764,null,12.565679,40.463667,null,37.09024,9.748917,null,45.943161,39.074208,null,38.963745,39.399872,null,37.09024,-18.665695,null,52.132633,45.1,null,38.963745,30.375321,null,46.603354,39.399872,null,14.058324,-25.274398,null,38.963745,63.397768,null,55.378051,20.593684,null,35.86166,26.02751,null,42.315407,46.603354,null,52.132633,-14.235004,null,52.132633,36.204824,null,46.603354,37.09024,null,45.943161,42.733883,null,35.86166,33.854721,null,30.375321,-25.274398,null,49.817492,41.87194,null,51.165691,39.074208,null,44.016521,45.943161,null,35.907757,1.352083,null,23.634501,51.165691,null,50.850346,-30.559482,null,23.634501,50.850346,null,35.86166,-18.665695,null,46.603354,48.669026,null,51.165691,44.016521,null,12.565679,51.165691,null,55.169438,51.165691,null,45.943161,48.669026,null,23.634501,56.130366,null,42.733883,45.943161,null,47.516231,51.919438,null,35.86166,35.126413,null,-35.675147,-9.189967,null,30.375321,15.783471,null,40.463667,37.09024,null,4.210484,37.09024,null,37.09024,1.352083,null,45.943161,52.132633,null,47.516231,46.818188,null,35.86166,49.815273,null,38.963745,48.019573,null,58.595272,56.879635,null,14.058324,4.210484,null,53.41291,55.378051,null,-0.789275,1.352083,null,38.963745,53.41291,null,35.907757,41.377491,null,-30.559482,-22.328474,null,51.919438,46.818188,null,55.169438,58.595272,null,35.86166,56.879635,null,12.565679,56.130366,null,15.870032,52.132633,null,35.86166,28.394857,null,41.87194,49.817492,null,51.165691,60.472024,null,38.963745,-25.274398,null,35.86166,7.946527,null,15.870032,40.463667,null,58.595272,61.92411,null,47.516231,49.817492,null,47.516231,48.669026,null,14.058324,23.424076,null,51.919438,44.016521,null,31.791702,51.165691,null,20.593684,35.86166,null,49.817492,52.132633,null,39.399872,31.791702,null,-40.900557,-25.274398,null,56.26392,40.463667,null,43.915886,51.165691,null,41.87194,38.963745,null,51.165691,48.379433,null,35.907757,38.963745,null,47.516231,45.943161,null,53.41291,-40.900557,null,-32.522779,-38.416097,null,37.09024,30.585164,null,48.669026,63.397768,null,35.907757,21.521757,null,38.963745,44.016521,null,38.963745,46.151241,null,36.204824,15.870032,null,51.165691,42.733883,null,38.963745,51.919438,null,37.09024,63.397768,null,51.919438,42.733883,null,30.375321,41.87194,null,56.26392,61.52401,null,30.375321,4.210484,null,56.26392,51.919438,null,47.162494,51.919438,null,14.058324,12.879721,null,31.046051,-0.228021,null,48.669026,46.603354,null,40.069099,61.52401,null,37.09024,14.058324,null,37.09024,-40.900557,null,30.375321,26.820553,null,15.870032,21.00789,null,35.86166,41.20438,null,37.09024,21.521757,null,45.1,44.016521,null,12.565679,46.603354,null,50.850346,41.87194,null,38.963745,50.850346,null,35.86166,40.339852,null,35.86166,18.109581,null,50.850346,46.818188,null,55.169438,51.919438,null,35.86166,46.862496,null,31.046051,51.919438,null,38.963745,61.92411,null,37.09024,4.570868,null,30.375321,4.570868,null,56.26392,50.850346,null,52.132633,60.472024,null,35.86166,42.315407,null,41.87194,47.162494,null,1.352083,-0.789275,null,51.165691,23.885942,null,47.411631,51.165691,null,55.169438,52.132633,null,49.817492,55.378051,null,63.397768,38.963745,null,48.019573,61.52401,null,44.016521,49.817492,null,42.315407,40.143105,null,55.378051,51.919438,null,-35.675147,37.09024,null,52.132633,46.151241,null,-30.559482,-2.1646,null,37.09024,53.41291,null,46.151241,47.516231,null,4.210484,36.204824,null,14.058324,45.943161,null,45.943161,40.463667,null,33.886917,30.585164,null,45.943161,47.516231,null,41.87194,45.943161,null,51.165691,47.411631,null,52.132633,42.733883,null,47.516231,46.151241,null,56.130366,50.850346,null,45.1,51.165691,null,55.378051,36.204824,null,45.943161,45.1,null,38.963745,35.126413,null,56.26392,-2.1646,null,22.396428,35.907757,null,46.603354,39.074208,null,36.204824,22.396428,null,40.463667,-6.314993,null,44.016521,50.850346,null,39.399872,50.850346,null,-14.235004,-23.442503,null,46.603354,63.397768,null,46.603354,31.791702,null,14.058324,61.52401,null,61.92411,58.595272,null,38.963745,17.570692,null,44.016521,51.165691,null,23.634501,12.865416,null,52.132633,23.634501,null,40.463667,-14.235004,null,45.943161,39.399872,null,51.165691,35.126413,null,51.919438,45.1,null,51.919438,46.151241,null,42.733883,39.074208,null,19.85627,46.603354,null,1.352083,12.565679,null,51.165691,30.585164,null,42.315407,40.463667,null,35.86166,21.512583,null,41.87194,41.0,null,-25.274398,28.394857,null,47.162494,41.87194,null,39.399872,45.943161,null,35.86166,33.886917,null,53.41291,36.204824,null,63.397768,37.09024,null,-35.675147,-14.235004,null,55.378051,-23.442503,null,46.603354,53.41291,null,46.818188,35.86166,null,40.463667,-30.559482,null,40.463667,50.850346,null,20.593684,51.919438,null,55.378051,60.472024,null,51.165691,31.046051,null,50.850346,53.41291,null,52.132633,15.552727,null,36.204824,12.879721,null,37.09024,20.593684,null,55.378051,-11.202692,null,63.397768,48.669026,null,63.397768,55.378051,null,51.919438,48.019573,null,26.820553,-6.369028,null,41.87194,50.850346,null,20.593684,12.862807,null,30.375321,12.52111,null,55.378051,1.352083,null,46.818188,51.165691,null,50.850346,28.0,null,35.86166,-3.373056,null,15.870032,55.378051,null,26.820553,-0.023559,null,-35.675147,35.86166,null,36.204824,50.850346,null,38.963745,56.26392,null,40.463667,9.145,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":0.5009242160126856},"lon":[104.195397,127.766922,null,5.291266,4.351721,null,-95.712891,-106.346771,null,104.195397,1.888334,null,104.195397,121.774017,null,104.195397,113.921327,null,104.195397,-102.552784,null,104.195397,-3.74922,null,104.195397,19.145136,null,108.277199,-95.712891,null,104.195397,78.96288,null,127.766922,104.195397,null,104.195397,105.318756,null,104.195397,4.351721,null,10.451526,1.888334,null,10.451526,19.145136,null,104.195397,101.975766,null,5.291266,1.888334,null,19.145136,10.451526,null,104.195397,16.354896,null,104.195397,45.079162,null,104.195397,12.56738,null,104.195397,108.277199,null,69.345116,-95.712891,null,104.195397,-71.542969,null,-70.162651,-95.712891,null,-106.346771,-95.712891,null,78.96288,-95.712891,null,23.881275,16.354896,null,121.774017,138.252924,null,104.195397,-51.92528,null,9.537499,1.888334,null,10.451526,14.550072,null,1.888334,-3.74922,null,100.992541,-95.712891,null,104.195397,103.819836,null,10.451526,12.56738,null,35.243322,-95.712891,null,108.277199,138.252924,null,10.451526,-3.74922,null,104.195397,53.847818,null,104.990963,-95.712891,null,35.243322,10.451526,null,10.451526,8.227512,null,104.195397,66.923684,null,104.195397,22.937506,null,10.451526,5.291266,null,-3.74922,1.888334,null,10.451526,15.472962,null,104.195397,114.109497,null,19.145136,15.472962,null,108.277199,10.451526,null,104.195397,30.802498,null,-3.74922,-8.224454,null,108.277199,127.766922,null,-7.09262,-3.74922,null,27.953389,105.318756,null,104.195397,21.824312,null,5.291266,19.145136,null,5.291266,12.56738,null,-3.74922,12.56738,null,-3.74922,-7.09262,null,104.195397,-78.183406,null,104.195397,-75.015152,null,10.451526,104.195397,null,104.195397,43.679291,null,43.356892,19.145136,null,46.869107,1.888334,null,16.354896,8.468946,null,78.96288,8.675277,null,21.824312,19.503304,null,15.472962,10.451526,null,5.291266,104.195397,null,10.451526,16.354896,null,69.345116,4.351721,null,10.451526,4.351721,null,5.291266,-3.74922,null,104.195397,-80.782127,null,-95.712891,104.195397,null,104.195397,-74.297333,null,104.195397,34.851612,null,104.195397,14.995463,null,104.195397,9.501785,null,21.824312,12.56738,null,95.955974,-95.712891,null,127.766922,114.109497,null,24.96676,10.451526,null,28.369885,24.96676,null,35.243322,28.369885,null,101.975766,35.243322,null,-95.712891,0.824782,null,-58.443832,-51.92528,null,10.451526,114.109497,null,1.888334,12.56738,null,104.195397,174.885971,null,138.252924,108.277199,null,4.351721,10.451526,null,104.195397,8.675277,null,4.351721,1.888334,null,104.195397,8.227512,null,104.195397,8.468946,null,-95.712891,4.351721,null,78.96288,53.847818,null,5.291266,-95.712891,null,19.145136,31.16558,null,104.195397,14.550072,null,104.195397,95.955974,null,24.96676,19.145136,null,101.975766,1.888334,null,114.109497,104.195397,null,4.351721,5.291266,null,127.766922,138.252924,null,14.550072,10.451526,null,104.195397,-8.24389,null,-8.224454,1.888334,null,10.451526,-3.435973,null,35.243322,12.56738,null,104.195397,90.3563,null,19.145136,5.291266,null,78.96288,5.291266,null,1.888334,10.451526,null,15.472962,19.145136,null,16.354896,19.145136,null,10.451526,9.501785,null,104.195397,-8.224454,null,35.243322,5.291266,null,35.243322,21.824312,null,108.277199,104.195397,null,-8.224454,15.472962,null,9.537499,12.56738,null,16.354896,25.748151,null,127.766922,-1.023194,null,104.195397,47.481766,null,35.243322,-3.74922,null,10.451526,19.503304,null,104.990963,104.195397,null,104.195397,-88.89653,null,19.145136,19.699024,null,-95.712891,-71.542969,null,104.195397,23.881275,null,78.96288,34.888822,null,-8.224454,-3.74922,null,104.195397,104.990963,null,5.291266,-8.24389,null,19.145136,19.503304,null,108.277199,-3.435973,null,104.195397,-63.616672,null,95.955974,127.766922,null,104.195397,19.503304,null,-7.09262,-3.435973,null,64.585262,71.276093,null,127.766922,108.277199,null,35.243322,1.888334,null,-95.712891,-80.782127,null,34.851612,-78.183406,null,16.354896,9.501785,null,24.96676,1.888334,null,104.195397,25.48583,null,9.501785,1.888334,null,1.888334,4.351721,null,12.56738,1.888334,null,108.277199,5.291266,null,15.472962,-95.712891,null,19.503304,23.881275,null,78.96288,37.906193,null,24.96676,12.56738,null,5.291266,-3.435973,null,5.291266,9.501785,null,19.145136,1.888334,null,104.990963,138.252924,null,1.888334,9.537499,null,108.277199,-3.74922,null,104.195397,34.888822,null,5.291266,15.472962,null,10.451526,9.537499,null,104.195397,69.345116,null,100.992541,15.472962,null,-95.712891,-3.435973,null,4.351721,6.129583,null,35.243322,24.96676,null,127.766922,-95.712891,null,4.351721,15.472962,null,5.291266,-8.224454,null,69.345116,5.291266,null,19.699024,14.550072,null,5.291266,25.013607,null,1.888334,5.291266,null,104.195397,25.748151,null,35.243322,19.503304,null,10.451526,-95.712891,null,114.109497,108.277199,null,19.145136,9.501785,null,19.145136,4.351721,null,138.252924,104.195397,null,19.145136,25.748151,null,19.145136,16.354896,null,-3.435973,-8.24389,null,15.472962,19.699024,null,104.195397,-83.753428,null,-106.346771,10.451526,null,-95.712891,133.775136,null,104.195397,24.96676,null,78.96288,12.56738,null,78.96288,10.451526,null,78.96288,100.992541,null,10.451526,24.96676,null,104.195397,17.873887,null,25.48583,10.451526,null,43.356892,5.291266,null,114.109497,-95.712891,null,69.345116,-3.435973,null,108.277199,19.145136,null,104.195397,30.217636,null,78.96288,1.888334,null,108.277199,1.888334,null,10.451526,-102.552784,null,19.145136,24.96676,null,5.291266,24.603189,null,19.145136,12.56738,null,-3.435973,10.451526,null,-8.224454,19.699024,null,103.819836,138.252924,null,5.291266,25.748151,null,104.195397,51.183884,null,35.243322,-3.435973,null,104.195397,35.243322,null,31.16558,19.699024,null,16.354896,10.451526,null,104.195397,80.771797,null,19.145136,-95.712891,null,1.888334,104.195397,null,5.291266,24.96676,null,12.56738,10.451526,null,1.888334,8.227512,null,10.451526,-8.24389,null,69.345116,25.48583,null,1.888334,24.96676,null,5.291266,16.354896,null,133.775136,-172.104629,null,35.243322,105.318756,null,5.291266,53.847818,null,78.96288,-106.346771,null,-1.023194,10.451526,null,10.451526,19.699024,null,108.277199,16.354896,null,104.195397,53.688046,null,101.975766,133.775136,null,35.243322,45.079162,null,35.243322,25.48583,null,-3.74922,-3.435973,null,31.16558,10.451526,null,78.96288,-3.435973,null,19.503304,24.96676,null,78.96288,133.775136,null,35.243322,43.679291,null,9.501785,8.468946,null,-3.74922,10.451526,null,9.537499,-3.74922,null,104.195397,-63.588653,null,43.356892,10.451526,null,1.888334,114.109497,null,21.824312,-3.74922,null,10.451526,14.995463,null,-95.712891,10.451526,null,24.603189,16.354896,null,19.145136,14.550072,null,104.195397,3.0,null,10.451526,23.881275,null,28.369885,19.699024,null,1.888334,19.145136,null,10.451526,-7.09262,null,-8.224454,19.145136,null,-102.552784,-70.162651,null,104.195397,-7.09262,null,10.451526,15.2,null,5.291266,35.243322,null,108.277199,104.990963,null,19.503304,10.451526,null,104.195397,17.228331,null,5.291266,23.881275,null,104.195397,-90.230759,null,127.766922,113.921327,null,35.243322,34.851612,null,35.243322,30.217636,null,103.819836,101.975766,null,100.992541,138.252924,null,69.345116,19.145136,null,-95.712891,19.145136,null,104.195397,-55.765835,null,5.291266,19.503304,null,10.451526,25.748151,null,104.195397,19.699024,null,113.921327,138.252924,null,69.345116,10.451526,null,114.109497,113.543873,null,104.195397,15.472962,null,19.699024,15.472962,null,101.975766,103.819836,null,9.501785,10.451526,null,5.291266,19.699024,null,9.537499,4.351721,null,104.195397,113.543873,null,103.819836,133.775136,null,-102.552784,-83.753428,null,-8.224454,10.451526,null,127.766922,104.990963,null,78.96288,101.975766,null,9.537499,19.699024,null,101.975766,114.109497,null,-8.224454,-3.435973,null,103.819836,114.109497,null,5.291266,14.550072,null,108.277199,-106.346771,null,-95.712891,138.252924,null,19.145136,23.881275,null,1.888334,-3.435973,null,-3.435973,1.888334,null,10.451526,-8.224454,null,12.56738,104.195397,null,15.472962,9.501785,null,101.975766,104.195397,null,-3.435973,5.291266,null,78.96288,69.345116,null,14.550072,19.503304,null,35.243322,15.472962,null,10.451526,-51.92528,null,104.195397,-86.241905,null,5.291266,6.129583,null,103.819836,104.195397,null,108.277199,-102.552784,null,-106.346771,0.824782,null,35.243322,14.550072,null,9.501785,16.354896,null,43.356892,19.503304,null,16.354896,24.603189,null,25.748151,16.354896,null,104.195397,36.238414,null,104.195397,24.15536,null,108.277199,12.56738,null,78.96288,138.252924,null,104.195397,15.2,null,34.851612,33.429859,null,-55.765835,-51.92528,null,127.766922,103.846656,null,35.243322,47.576927,null,35.243322,17.228331,null,133.775136,174.885971,null,19.145136,24.603189,null,-8.24389,37.906193,null,15.472962,1.888334,null,78.96288,90.3563,null,104.195397,-70.162651,null,31.16558,24.96676,null,43.356892,15.472962,null,69.345116,-106.346771,null,104.195397,64.585262,null,5.291266,21.824312,null,108.277199,113.921327,null,-8.24389,29.154857,null,104.195397,-61.222503,null,104.195397,-14.452362,null,-3.74922,-71.542969,null,19.145136,-3.74922,null,4.351721,-3.74922,null,-7.09262,-8.224454,null,5.291266,8.227512,null,19.145136,-3.435973,null,100.992541,10.451526,null,10.451526,35.243322,null,12.56738,8.227512,null,69.345116,1.888334,null,-95.712891,66.923684,null,-95.712891,5.291266,null,-95.712891,114.109497,null,15.472962,-3.74922,null,4.351721,19.145136,null,108.277199,4.351721,null,-95.712891,-70.162651,null,35.243322,-7.09262,null,78.96288,45.079162,null,10.451526,6.129583,null,104.195397,37.906193,null,121.774017,78.96288,null,24.96676,35.243322,null,35.243322,19.699024,null,104.195397,46.869107,null,114.109497,10.451526,null,104.195397,31.16558,null,114.109497,138.252924,null,9.501785,-8.24389,null,15.472962,14.550072,null,100.992541,133.775136,null,19.699024,19.503304,null,19.145136,104.195397,null,5.291266,-106.346771,null,4.351721,9.537499,null,19.699024,10.451526,null,21.824312,31.16558,null,34.851612,-55.765835,null,10.451526,30.802498,null,12.56738,-3.74922,null,43.356892,4.351721,null,1.888334,14.550072,null,104.195397,27.953389,null,23.881275,9.501785,null,-3.74922,104.195397,null,-3.435973,114.109497,null,78.96288,55.923255,null,24.96676,4.351721,null,35.243322,43.356892,null,104.195397,-66.58973,null,-3.435973,22.937506,null,1.888334,174.885971,null,15.472962,4.351721,null,19.699024,19.145136,null,22.937506,18.49041,null,104.990963,-3.74922,null,-95.712891,-83.753428,null,24.96676,21.824312,null,35.243322,-8.224454,null,-95.712891,35.529562,null,5.291266,15.2,null,35.243322,69.345116,null,1.888334,-8.224454,null,108.277199,133.775136,null,35.243322,16.354896,null,-3.435973,78.96288,null,104.195397,50.55096,null,43.356892,1.888334,null,5.291266,-51.92528,null,5.291266,138.252924,null,1.888334,-95.712891,null,24.96676,25.48583,null,104.195397,35.862285,null,69.345116,133.775136,null,15.472962,12.56738,null,10.451526,21.824312,null,21.005859,24.96676,null,127.766922,103.819836,null,-102.552784,10.451526,null,4.351721,22.937506,null,-102.552784,4.351721,null,104.195397,35.529562,null,1.888334,19.699024,null,10.451526,21.005859,null,104.990963,10.451526,null,23.881275,10.451526,null,24.96676,19.699024,null,-102.552784,-106.346771,null,25.48583,24.96676,null,14.550072,19.145136,null,104.195397,33.429859,null,-71.542969,-75.015152,null,69.345116,-90.230759,null,-3.74922,-95.712891,null,101.975766,-95.712891,null,-95.712891,103.819836,null,24.96676,5.291266,null,14.550072,8.227512,null,104.195397,6.129583,null,35.243322,66.923684,null,25.013607,24.603189,null,108.277199,101.975766,null,-8.24389,-3.435973,null,113.921327,103.819836,null,35.243322,-8.24389,null,127.766922,64.585262,null,22.937506,24.684866,null,19.145136,8.227512,null,23.881275,25.013607,null,104.195397,24.603189,null,104.990963,-106.346771,null,100.992541,5.291266,null,104.195397,84.124008,null,12.56738,15.472962,null,10.451526,8.468946,null,35.243322,133.775136,null,104.195397,-1.023194,null,100.992541,-3.74922,null,25.013607,25.748151,null,14.550072,15.472962,null,14.550072,19.699024,null,108.277199,53.847818,null,19.145136,21.005859,null,-7.09262,10.451526,null,78.96288,104.195397,null,15.472962,5.291266,null,-8.224454,-7.09262,null,174.885971,133.775136,null,9.501785,-3.74922,null,17.679076,10.451526,null,12.56738,35.243322,null,10.451526,31.16558,null,127.766922,35.243322,null,14.550072,24.96676,null,-8.24389,174.885971,null,-55.765835,-63.616672,null,-95.712891,36.238414,null,19.699024,16.354896,null,127.766922,-77.781167,null,35.243322,21.005859,null,35.243322,14.995463,null,138.252924,100.992541,null,10.451526,25.48583,null,35.243322,19.145136,null,-95.712891,16.354896,null,19.145136,25.48583,null,69.345116,12.56738,null,9.501785,105.318756,null,69.345116,101.975766,null,9.501785,19.145136,null,19.503304,19.145136,null,108.277199,121.774017,null,34.851612,15.827659,null,19.699024,1.888334,null,45.038189,105.318756,null,-95.712891,108.277199,null,-95.712891,174.885971,null,69.345116,30.802498,null,100.992541,-10.940835,null,104.195397,74.766098,null,-95.712891,-77.781167,null,15.2,21.005859,null,104.990963,1.888334,null,4.351721,12.56738,null,35.243322,4.351721,null,104.195397,127.510093,null,104.195397,-77.297508,null,4.351721,8.227512,null,23.881275,19.145136,null,104.195397,103.846656,null,34.851612,19.145136,null,35.243322,25.748151,null,-95.712891,-74.297333,null,69.345116,-74.297333,null,9.501785,4.351721,null,5.291266,8.468946,null,104.195397,43.356892,null,12.56738,19.503304,null,103.819836,113.921327,null,10.451526,45.079162,null,28.369885,10.451526,null,23.881275,5.291266,null,15.472962,-3.435973,null,16.354896,35.243322,null,66.923684,105.318756,null,21.005859,15.472962,null,43.356892,47.576927,null,-3.435973,19.145136,null,-71.542969,-95.712891,null,5.291266,14.995463,null,22.937506,24.15536,null,-95.712891,-8.24389,null,14.995463,14.550072,null,101.975766,138.252924,null,108.277199,24.96676,null,24.96676,-3.74922,null,9.537499,36.238414,null,24.96676,14.550072,null,12.56738,24.96676,null,10.451526,28.369885,null,5.291266,25.48583,null,14.550072,14.995463,null,-106.346771,4.351721,null,15.2,10.451526,null,-3.435973,138.252924,null,24.96676,15.2,null,35.243322,33.429859,null,9.501785,24.15536,null,114.109497,127.766922,null,1.888334,21.824312,null,138.252924,114.109497,null,-3.74922,143.95555,null,21.005859,4.351721,null,-8.224454,4.351721,null,-51.92528,-58.443832,null,1.888334,16.354896,null,1.888334,-7.09262,null,108.277199,105.318756,null,25.748151,25.013607,null,35.243322,-3.996166,null,21.005859,10.451526,null,-102.552784,-85.207229,null,5.291266,-102.552784,null,-3.74922,-51.92528,null,24.96676,-8.224454,null,10.451526,33.429859,null,19.145136,15.2,null,19.145136,14.995463,null,25.48583,21.824312,null,102.495496,1.888334,null,103.819836,104.990963,null,10.451526,36.238414,null,43.356892,-3.74922,null,104.195397,55.923255,null,12.56738,20.0,null,133.775136,84.124008,null,19.503304,12.56738,null,-8.224454,24.96676,null,104.195397,9.537499,null,-8.24389,138.252924,null,16.354896,-95.712891,null,-71.542969,-51.92528,null,-3.435973,-58.443832,null,1.888334,-8.24389,null,8.227512,104.195397,null,-3.74922,22.937506,null,-3.74922,4.351721,null,78.96288,19.145136,null,-3.435973,8.468946,null,10.451526,34.851612,null,4.351721,-8.24389,null,5.291266,48.516388,null,138.252924,121.774017,null,-95.712891,78.96288,null,-3.435973,17.873887,null,16.354896,19.699024,null,16.354896,-3.435973,null,19.145136,66.923684,null,30.802498,34.888822,null,12.56738,4.351721,null,78.96288,30.217636,null,69.345116,-69.968338,null,-3.435973,103.819836,null,8.227512,10.451526,null,4.351721,3.0,null,104.195397,29.918886,null,100.992541,-3.435973,null,30.802498,37.906193,null,-71.542969,104.195397,null,138.252924,4.351721,null,35.243322,9.501785,null,-3.74922,40.489673,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,46.603354,-11.202692,-0.228021,51.165691,-22.95764,37.09024,17.060816,40.143105,42.315407,33.854721,46.818188,55.378051,-38.416097,-35.675147,-23.442503,-32.522779,-25.274398,50.850346,-22.328474,42.733883,56.130366,7.873054,35.86166,-10.447525,-17.713371,7.946527,22.396428,-0.789275,31.046051,41.87194,7.539989,36.204824,35.907757,4.210484,35.937496,46.862496,-0.522778,52.132633,-20.904305,-15.376706,-40.900557,-29.040835,8.537981,-6.314993,-9.189967,12.879721,25.354826,20.593684,1.352083,-30.559482,15.870032,23.424076,26.820553,47.516231,28.0,23.685,-16.290154,43.915886,-14.235004,53.709807,15.454166,4.570868,45.1,35.126413,49.817492,56.26392,18.735693,58.595272,61.92411,-17.679742,13.443182,39.074208,47.162494,64.963051,32.427908,33.223191,53.41291,48.019573,30.585164,29.31166,19.85627,56.879635,55.169438,49.815273,17.570692,21.00789,-20.348404,23.634501,47.411631,42.708678,31.791702,12.865416,9.081999,60.472024,30.375321,51.919438,39.399872,45.943161,61.52401,12.984305,43.94236,23.885942,14.497401,44.016521,-4.679574,48.669026,14.058324,46.151241,40.463667,63.397768,33.886917,38.963745,1.373333,48.379433,41.608635,26.02751,21.512583,40.069099,13.193887,12.262776,7.369722,16.5388,6.611111,-2.1646,9.30769,1.650801,-0.803689,36.140751,18.971187,-0.023559,26.3351,-18.766947,17.607789,-1.940278,8.460555,-19.015438,12.862807,15.552727,12.565679,25.03428,32.3078,9.748917,-1.831239,13.794185,4.860416,18.109581,6.428055,7.131474,17.357822,10.691803,4.535277,41.0,21.916221,19.3133,21.521757,15.783471,12.16957,46.946947,33.0,27.514162,-9.64571,-3.373056,-11.6455,15.414999,9.145,15.179384,11.825138,31.952162,9.945587,15.199999,40.339852,41.20438,-29.609988,22.198745,-13.254308,3.202778,-18.665695,28.394857,12.52111,11.803749,-8.874217,13.909444,0.18636,5.152149,7.862684,3.919305,34.802075,38.861034,8.619543,-21.178986,38.969719,21.694025,-6.369028,12.238333,41.377491,6.42375,-13.759029,-13.133897,71.706936,61.892635,-26.522503,-14.28522,-3.370417,13.444304,-13.768752,18.04248,17.897476,-9.2,17.189877,7.425554,17.664332,7.51498,12.20189,-21.236736,-49.280366,-51.796253,18.420695,18.218785,-19.054445,10.961632,-7.109535,-24.376515,16.742498,-15.965,-90.0,-12.164165,24.215527,-6.343194],"lon":[1.601554,1.888334,17.873887,15.827659,10.451526,18.49041,-95.712891,-61.796428,47.576927,43.356892,35.862285,8.227512,-3.435973,-63.616672,-71.542969,-58.443832,-55.765835,133.775136,4.351721,24.684866,25.48583,-106.346771,80.771797,104.195397,105.690449,178.065033,-1.023194,114.109497,113.921327,34.851612,12.56738,-5.54708,138.252924,127.766922,101.975766,14.375416,103.846656,166.931503,5.291266,165.618042,166.959158,174.885971,167.954712,-80.782127,143.95555,-75.015152,121.774017,51.183884,78.96288,103.819836,22.937506,100.992541,53.847818,30.802498,14.550072,3.0,90.3563,-63.588653,17.679076,-51.92528,27.953389,18.732207,-74.297333,15.2,33.429859,15.472962,9.501785,-70.162651,25.013607,25.748151,-149.406843,-15.310139,21.824312,19.503304,-19.020835,53.688046,43.679291,-8.24389,66.923684,36.238414,47.481766,102.495496,24.603189,23.881275,6.129583,-3.996166,-10.940835,57.552152,-102.552784,28.369885,19.37439,-7.09262,-85.207229,8.675277,8.468946,69.345116,19.145136,-8.224454,24.96676,105.318756,-61.287228,12.457777,45.079162,-14.452362,21.005859,55.491977,19.699024,108.277199,14.995463,-3.74922,16.354896,9.537499,35.243322,32.290275,31.16558,21.745275,50.55096,55.923255,45.038189,-59.543198,-61.604171,12.354722,-23.0418,20.939444,24.15536,2.315834,10.267895,11.609444,-5.353585,-72.285215,37.906193,17.228331,46.869107,8.081666,29.873888,-11.779889,29.154857,30.217636,48.516388,104.990963,-77.39628,-64.7505,-83.753428,-78.183406,-88.89653,-58.93018,-77.297508,-9.429499,171.184478,-62.782998,-61.222503,114.727669,20.0,95.955974,-81.2546,-77.781167,-90.230759,-68.990021,-56.32509,65.0,90.433601,160.156194,29.918886,43.3333,-61.370976,40.489673,39.782334,42.590275,35.233154,-9.696645,-86.241905,127.510093,74.766098,28.233608,113.543873,34.301525,73.22068,35.529562,84.124008,-69.968338,-15.180413,125.727,-60.978893,6.613081,46.199616,30.217636,-56.027783,38.996815,71.276093,0.824782,-175.198242,59.556278,-71.797928,34.888822,-1.561593,64.585262,-66.58973,-172.104629,27.849332,-42.604303,-6.911806,31.465866,-170.70444,-168.734039,144.793731,-177.156097,-63.05483,-62.83055,-171.833333,-88.49765,150.550812,145.94351,134.58252,-68.262383,-159.777671,69.348557,-59.523613,-64.639968,-63.043653,-169.867233,-169.09022,179.194167,-128.324001,-62.187366,-5.7089,0.0,96.870956,-12.885834,71.876519],"marker":{"color":"red","opacity":0.8,"size":[7.2,20.0,20.0,8.8,20.0,20.0,20.0,10.0,19.6,20.0,14.8,20.0,20.0,19.2,20.0,15.6,20.0,20.0,20.0,12.0,20.0,20.0,10.8,20.0,0.8,16.4,20.0,20.0,20.0,20.0,20.0,12.4,20.0,20.0,20.0,20.0,12.8,1.6,20.0,11.6,6.0,20.0,0.8,19.6,8.8,15.6,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,11.2,11.2,14.8,20.0,20.0,20.0,6.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,12.4,5.6,20.0,20.0,17.2,6.8,13.2,20.0,20.0,16.4,20.0,9.6,20.0,20.0,20.0,9.2,9.2,20.0,20.0,20.0,16.8,20.0,12.8,14.0,20.0,20.0,20.0,20.0,20.0,20.0,4.4,5.2,20.0,16.4,20.0,10.8,20.0,20.0,20.0,20.0,20.0,20.0,20.0,10.4,20.0,20.0,20.0,20.0,18.0,14.8,5.2,10.0,3.6,5.6,12.0,8.4,5.6,8.8,7.2,5.6,20.0,11.2,16.0,6.4,10.4,4.4,8.4,8.4,6.0,20.0,8.0,6.4,14.4,16.8,16.4,13.2,12.0,6.8,4.0,4.0,10.0,18.0,12.4,14.0,4.8,11.2,20.0,6.4,1.2,6.4,5.2,4.0,6.0,4.0,3.6,9.6,3.2,6.8,2.4,7.6,10.4,0.4,18.4,2.8,12.8,5.2,10.0,12.0,8.0,6.4,3.6,4.0,4.4,3.2,6.4,2.8,5.6,2.4,5.6,8.8,3.2,6.8,3.6,16.0,16.8,18.0,11.2,3.6,11.2,6.8,7.6,4.0,0.8,2.0,5.2,1.2,4.0,2.8,0.4,6.4,2.4,2.4,2.8,1.2,3.6,1.2,0.4,6.4,1.6,0.8,0.8,2.0,0.4,0.8,0.8,0.8,0.4,0.4,0.4]},"mode":"markers","name":"2022","text":["AND\u003cbr\u003eConex\u00f5es: 18","FRA\u003cbr\u003eConex\u00f5es: 253","AGO\u003cbr\u003eConex\u00f5es: 54","COG\u003cbr\u003eConex\u00f5es: 22","DEU\u003cbr\u003eConex\u00f5es: 264","NAM\u003cbr\u003eConex\u00f5es: 64","USA\u003cbr\u003eConex\u00f5es: 271","ATG\u003cbr\u003eConex\u00f5es: 25","AZE\u003cbr\u003eConex\u00f5es: 49","GEO\u003cbr\u003eConex\u00f5es: 65","LBN\u003cbr\u003eConex\u00f5es: 37","CHE\u003cbr\u003eConex\u00f5es: 202","GBR\u003cbr\u003eConex\u00f5es: 246","ARG\u003cbr\u003eConex\u00f5es: 48","CHL\u003cbr\u003eConex\u00f5es: 73","PRY\u003cbr\u003eConex\u00f5es: 39","URY\u003cbr\u003eConex\u00f5es: 51","AUS\u003cbr\u003eConex\u00f5es: 164","BEL\u003cbr\u003eConex\u00f5es: 217","BWA\u003cbr\u003eConex\u00f5es: 30","BGR\u003cbr\u003eConex\u00f5es: 116","CAN\u003cbr\u003eConex\u00f5es: 204","LKA\u003cbr\u003eConex\u00f5es: 27","CHN\u003cbr\u003eConex\u00f5es: 274","CXR\u003cbr\u003eConex\u00f5es: 2","FJI\u003cbr\u003eConex\u00f5es: 41","GHA\u003cbr\u003eConex\u00f5es: 50","HKG\u003cbr\u003eConex\u00f5es: 172","IDN\u003cbr\u003eConex\u00f5es: 95","ISR\u003cbr\u003eConex\u00f5es: 109","ITA\u003cbr\u003eConex\u00f5es: 213","CIV\u003cbr\u003eConex\u00f5es: 31","JPN\u003cbr\u003eConex\u00f5es: 154","KOR\u003cbr\u003eConex\u00f5es: 204","MYS\u003cbr\u003eConex\u00f5es: 93","MLT\u003cbr\u003eConex\u00f5es: 59","MNG\u003cbr\u003eConex\u00f5es: 32","NRU\u003cbr\u003eConex\u00f5es: 4","NLD\u003cbr\u003eConex\u00f5es: 233","NCL\u003cbr\u003eConex\u00f5es: 29","VUT\u003cbr\u003eConex\u00f5es: 15","NZL\u003cbr\u003eConex\u00f5es: 114","NFK\u003cbr\u003eConex\u00f5es: 2","PAN\u003cbr\u003eConex\u00f5es: 49","PNG\u003cbr\u003eConex\u00f5es: 22","PER\u003cbr\u003eConex\u00f5es: 39","PHL\u003cbr\u003eConex\u00f5es: 83","QAT\u003cbr\u003eConex\u00f5es: 51","IND\u003cbr\u003eConex\u00f5es: 215","SGP\u003cbr\u003eConex\u00f5es: 146","ZAF\u003cbr\u003eConex\u00f5es: 158","THA\u003cbr\u003eConex\u00f5es: 166","ARE\u003cbr\u003eConex\u00f5es: 75","EGY\u003cbr\u003eConex\u00f5es: 63","AUT\u003cbr\u003eConex\u00f5es: 218","DZA\u003cbr\u003eConex\u00f5es: 28","BGD\u003cbr\u003eConex\u00f5es: 28","BOL\u003cbr\u003eConex\u00f5es: 37","BIH\u003cbr\u003eConex\u00f5es: 56","BRA\u003cbr\u003eConex\u00f5es: 158","BLR\u003cbr\u003eConex\u00f5es: 54","TCD\u003cbr\u003eConex\u00f5es: 15","COL\u003cbr\u003eConex\u00f5es: 85","HRV\u003cbr\u003eConex\u00f5es: 100","CYP\u003cbr\u003eConex\u00f5es: 59","CZE\u003cbr\u003eConex\u00f5es: 172","DNK\u003cbr\u003eConex\u00f5es: 201","DOM\u003cbr\u003eConex\u00f5es: 62","EST\u003cbr\u003eConex\u00f5es: 107","FIN\u003cbr\u003eConex\u00f5es: 143","PYF\u003cbr\u003eConex\u00f5es: 31","GMB\u003cbr\u003eConex\u00f5es: 14","GRC\u003cbr\u003eConex\u00f5es: 131","HUN\u003cbr\u003eConex\u00f5es: 141","ISL\u003cbr\u003eConex\u00f5es: 43","IRN\u003cbr\u003eConex\u00f5es: 17","IRQ\u003cbr\u003eConex\u00f5es: 33","IRL\u003cbr\u003eConex\u00f5es: 117","KAZ\u003cbr\u003eConex\u00f5es: 68","JOR\u003cbr\u003eConex\u00f5es: 41","KWT\u003cbr\u003eConex\u00f5es: 55","LAO\u003cbr\u003eConex\u00f5es: 24","LVA\u003cbr\u003eConex\u00f5es: 98","LTU\u003cbr\u003eConex\u00f5es: 120","LUX\u003cbr\u003eConex\u00f5es: 85","MLI\u003cbr\u003eConex\u00f5es: 23","MRT\u003cbr\u003eConex\u00f5es: 23","MUS\u003cbr\u003eConex\u00f5es: 54","MEX\u003cbr\u003eConex\u00f5es: 81","MDA\u003cbr\u003eConex\u00f5es: 58","MNE\u003cbr\u003eConex\u00f5es: 42","MAR\u003cbr\u003eConex\u00f5es: 97","NIC\u003cbr\u003eConex\u00f5es: 32","NGA\u003cbr\u003eConex\u00f5es: 35","NOR\u003cbr\u003eConex\u00f5es: 118","PAK\u003cbr\u003eConex\u00f5es: 137","POL\u003cbr\u003eConex\u00f5es: 193","PRT\u003cbr\u003eConex\u00f5es: 189","ROU\u003cbr\u003eConex\u00f5es: 127","RUS\u003cbr\u003eConex\u00f5es: 53","VCT\u003cbr\u003eConex\u00f5es: 11","SMR\u003cbr\u003eConex\u00f5es: 13","SAU\u003cbr\u003eConex\u00f5es: 64","SEN\u003cbr\u003eConex\u00f5es: 41","SRB\u003cbr\u003eConex\u00f5es: 93","SYC\u003cbr\u003eConex\u00f5es: 27","SVK\u003cbr\u003eConex\u00f5es: 121","VNM\u003cbr\u003eConex\u00f5es: 115","SVN\u003cbr\u003eConex\u00f5es: 124","ESP\u003cbr\u003eConex\u00f5es: 204","SWE\u003cbr\u003eConex\u00f5es: 177","TUN\u003cbr\u003eConex\u00f5es: 74","TUR\u003cbr\u003eConex\u00f5es: 245","UGA\u003cbr\u003eConex\u00f5es: 26","UKR\u003cbr\u003eConex\u00f5es: 114","MKD\u003cbr\u003eConex\u00f5es: 61","BHR\u003cbr\u003eConex\u00f5es: 57","OMN\u003cbr\u003eConex\u00f5es: 54","ARM\u003cbr\u003eConex\u00f5es: 45","BRB\u003cbr\u003eConex\u00f5es: 37","GRD\u003cbr\u003eConex\u00f5es: 13","CMR\u003cbr\u003eConex\u00f5es: 25","CPV\u003cbr\u003eConex\u00f5es: 9","CAF\u003cbr\u003eConex\u00f5es: 14","COD\u003cbr\u003eConex\u00f5es: 30","BEN\u003cbr\u003eConex\u00f5es: 21","GNQ\u003cbr\u003eConex\u00f5es: 14","GAB\u003cbr\u003eConex\u00f5es: 22","GIB\u003cbr\u003eConex\u00f5es: 18","HTI\u003cbr\u003eConex\u00f5es: 14","KEN\u003cbr\u003eConex\u00f5es: 83","LBY\u003cbr\u003eConex\u00f5es: 28","MDG\u003cbr\u003eConex\u00f5es: 40","NER\u003cbr\u003eConex\u00f5es: 16","RWA\u003cbr\u003eConex\u00f5es: 26","SLE\u003cbr\u003eConex\u00f5es: 11","ZWE\u003cbr\u003eConex\u00f5es: 21","SDN\u003cbr\u003eConex\u00f5es: 21","YEM\u003cbr\u003eConex\u00f5es: 15","KHM\u003cbr\u003eConex\u00f5es: 55","BHS\u003cbr\u003eConex\u00f5es: 20","BMU\u003cbr\u003eConex\u00f5es: 16","CRI\u003cbr\u003eConex\u00f5es: 36","ECU\u003cbr\u003eConex\u00f5es: 42","SLV\u003cbr\u003eConex\u00f5es: 41","GUY\u003cbr\u003eConex\u00f5es: 33","JAM\u003cbr\u003eConex\u00f5es: 30","LBR\u003cbr\u003eConex\u00f5es: 17","MHL\u003cbr\u003eConex\u00f5es: 10","KNA\u003cbr\u003eConex\u00f5es: 10","TTO\u003cbr\u003eConex\u00f5es: 25","BRN\u003cbr\u003eConex\u00f5es: 45","ALB\u003cbr\u003eConex\u00f5es: 31","MMR\u003cbr\u003eConex\u00f5es: 35","CYM\u003cbr\u003eConex\u00f5es: 12","CUB\u003cbr\u003eConex\u00f5es: 28","GTM\u003cbr\u003eConex\u00f5es: 51","CUW\u003cbr\u003eConex\u00f5es: 16","SPM\u003cbr\u003eConex\u00f5es: 3","AFG\u003cbr\u003eConex\u00f5es: 16","BTN\u003cbr\u003eConex\u00f5es: 13","SLB\u003cbr\u003eConex\u00f5es: 10","BDI\u003cbr\u003eConex\u00f5es: 15","COM\u003cbr\u003eConex\u00f5es: 10","DMA\u003cbr\u003eConex\u00f5es: 9","ETH\u003cbr\u003eConex\u00f5es: 24","ERI\u003cbr\u003eConex\u00f5es: 8","DJI\u003cbr\u003eConex\u00f5es: 17","PSE\u003cbr\u003eConex\u00f5es: 6","GIN\u003cbr\u003eConex\u00f5es: 19","HND\u003cbr\u003eConex\u00f5es: 26","PRK\u003cbr\u003eConex\u00f5es: 1","KGZ\u003cbr\u003eConex\u00f5es: 46","LSO\u003cbr\u003eConex\u00f5es: 7","MAC\u003cbr\u003eConex\u00f5es: 32","MWI\u003cbr\u003eConex\u00f5es: 13","MDV\u003cbr\u003eConex\u00f5es: 25","MOZ\u003cbr\u003eConex\u00f5es: 30","NPL\u003cbr\u003eConex\u00f5es: 20","ABW\u003cbr\u003eConex\u00f5es: 16","GNB\u003cbr\u003eConex\u00f5es: 9","TLS\u003cbr\u003eConex\u00f5es: 10","LCA\u003cbr\u003eConex\u00f5es: 11","STP\u003cbr\u003eConex\u00f5es: 8","SOM\u003cbr\u003eConex\u00f5es: 16","SSD\u003cbr\u003eConex\u00f5es: 7","SUR\u003cbr\u003eConex\u00f5es: 14","SYR\u003cbr\u003eConex\u00f5es: 6","TJK\u003cbr\u003eConex\u00f5es: 14","TGO\u003cbr\u003eConex\u00f5es: 22","TON\u003cbr\u003eConex\u00f5es: 8","TKM\u003cbr\u003eConex\u00f5es: 17","TCA\u003cbr\u003eConex\u00f5es: 9","TZA\u003cbr\u003eConex\u00f5es: 40","BFA\u003cbr\u003eConex\u00f5es: 42","UZB\u003cbr\u003eConex\u00f5es: 45","VEN\u003cbr\u003eConex\u00f5es: 28","WSM\u003cbr\u003eConex\u00f5es: 9","ZMB\u003cbr\u003eConex\u00f5es: 28","GRL\u003cbr\u003eConex\u00f5es: 17","FRO\u003cbr\u003eConex\u00f5es: 19","SWZ\u003cbr\u003eConex\u00f5es: 10","ASM\u003cbr\u003eConex\u00f5es: 2","KIR\u003cbr\u003eConex\u00f5es: 5","GUM\u003cbr\u003eConex\u00f5es: 13","WLF\u003cbr\u003eConex\u00f5es: 3","SXM\u003cbr\u003eConex\u00f5es: 10","BLM\u003cbr\u003eConex\u00f5es: 7","TKL\u003cbr\u003eConex\u00f5es: 1","BLZ\u003cbr\u003eConex\u00f5es: 16","FSM\u003cbr\u003eConex\u00f5es: 6","MNP\u003cbr\u003eConex\u00f5es: 6","PLW\u003cbr\u003eConex\u00f5es: 7","BES\u003cbr\u003eConex\u00f5es: 3","COK\u003cbr\u003eConex\u00f5es: 9","ATF\u003cbr\u003eConex\u00f5es: 3","FLK\u003cbr\u003eConex\u00f5es: 1","VGB\u003cbr\u003eConex\u00f5es: 16","AIA\u003cbr\u003eConex\u00f5es: 4","NIU\u003cbr\u003eConex\u00f5es: 2","UMI\u003cbr\u003eConex\u00f5es: 2","TUV\u003cbr\u003eConex\u00f5es: 5","PCN\u003cbr\u003eConex\u00f5es: 1","MSR\u003cbr\u003eConex\u00f5es: 2","SHN\u003cbr\u003eConex\u00f5es: 2","ATA\u003cbr\u003eConex\u00f5es: 2","CCK\u003cbr\u003eConex\u00f5es: 1","ESH\u003cbr\u003eConex\u00f5es: 1","IOT\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,37.09024,null,35.86166,36.204824,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.6759832450000001},"lon":[104.195397,-95.712891,null,104.195397,138.252924,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[35.86166,51.165691,null,23.634501,37.09024,null,35.86166,55.378051,null,35.86166,52.132633,null,35.86166,-25.274398,null,35.86166,56.130366,null,37.09024,23.634501,null,35.86166,35.907757,null,35.86166,46.603354,null,52.132633,51.165691,null,-18.766947,46.603354,null,35.86166,23.634501,null,37.09024,56.130366,null,35.86166,15.870032,null,35.86166,51.919438,null,20.593684,23.424076,null,35.86166,-0.789275,null,52.132633,50.850346,null,35.86166,12.879721,null,35.86166,40.463667,null,35.86166,23.885942,null,35.86166,50.850346,null,35.86166,20.593684,null,35.86166,14.058324,null,35.86166,4.210484,null,14.058324,37.09024,null,35.86166,41.87194,null,51.165691,46.603354,null,51.919438,51.165691,null,35.86166,61.52401,null,51.165691,51.919438,null,38.963745,37.09024,null,52.132633,46.603354,null,12.565679,37.09024,null,35.86166,63.397768,null,18.735693,37.09024,null,35.86166,-14.235004,null,35.86166,23.424076,null,35.86166,-35.675147,null,35.86166,-30.559482,null,20.593684,26.02751,null,51.165691,47.516231,null,15.870032,37.09024,null,30.375321,37.09024,null,56.130366,37.09024,null,51.165691,41.87194,null,20.593684,37.09024,null,51.165691,49.817492,null,14.058324,51.165691,null,55.378051,31.791702,null,35.86166,1.352083,null,51.165691,40.463667,null,35.86166,22.396428,null,51.165691,46.818188,null,31.791702,40.463667,null,35.86166,39.074208,null,40.463667,39.399872,null,14.058324,36.204824,null,35.86166,56.26392,null,51.919438,49.817492,null,14.058324,35.907757,null,42.315407,51.919438,null,-18.766947,35.86166,null,38.963745,51.165691,null,35.86166,31.046051,null,38.963745,33.223191,null,35.86166,4.570868,null,51.165691,52.132633,null,35.86166,48.019573,null,40.463667,41.87194,null,35.86166,46.151241,null,15.870032,51.165691,null,49.817492,51.165691,null,4.210484,38.963745,null,21.916221,37.09024,null,40.463667,46.603354,null,35.86166,9.081999,null,35.86166,33.223191,null,63.397768,60.472024,null,40.463667,31.791702,null,35.86166,-0.023559,null,35.86166,60.472024,null,51.165691,50.850346,null,49.817492,51.919438,null,35.86166,23.685,null,35.86166,-9.189967,null,35.86166,-40.900557,null,39.399872,40.463667,null,46.603354,41.87194,null,52.132633,37.09024,null,38.963745,52.132633,null,51.919438,52.132633,null,45.943161,51.919438,null,52.132633,40.463667,null,38.963745,47.411631,null,38.963745,41.87194,null,46.603354,51.165691,null,51.165691,55.378051,null,51.165691,63.397768,null,50.850346,51.165691,null,35.86166,12.565679,null,36.204824,14.058324,null,35.86166,39.399872,null,4.210484,46.603354,null,35.86166,46.818188,null,52.132633,41.87194,null,46.603354,40.463667,null,41.87194,46.603354,null,21.916221,35.907757,null,35.86166,53.41291,null,30.375321,51.165691,null,52.132633,51.919438,null,50.850346,52.132633,null,39.399872,49.817492,null,51.919438,46.603354,null,38.963745,40.463667,null,45.943161,51.165691,null,51.919438,47.162494,null,-23.442503,-14.235004,null,35.86166,49.817492,null,63.397768,51.919438,null,49.817492,37.09024,null,51.919438,48.669026,null,35.86166,61.92411,null,46.603354,50.850346,null,35.86166,45.943161,null,35.86166,47.162494,null,14.058324,41.87194,null,38.963745,45.943161,null,37.09024,18.735693,null,14.058324,40.463667,null,35.86166,41.20438,null,22.396428,35.86166,null,50.850346,46.603354,null,35.86166,26.820553,null,35.86166,47.516231,null,35.86166,18.735693,null,35.86166,9.748917,null,36.204824,35.86166,null,39.399872,28.0,null,35.86166,55.169438,null,63.397768,56.26392,null,51.919438,48.379433,null,63.397768,61.92411,null,31.791702,55.378051,null,35.86166,6.42375,null,35.86166,48.379433,null,35.86166,48.669026,null,47.411631,45.943161,null,14.058324,55.378051,null,35.86166,-32.522779,null,20.593684,14.497401,null,20.593684,7.873054,null,30.375321,52.132633,null,14.058324,46.603354,null,37.09024,50.850346,null,35.86166,30.585164,null,14.058324,51.919438,null,51.919438,56.26392,null,14.058324,52.132633,null,42.733883,51.165691,null,12.565679,36.204824,null,45.943161,46.603354,null,12.565679,-25.274398,null,35.86166,7.946527,null,41.377491,38.861034,null,22.396428,37.09024,null,47.162494,51.165691,null,15.870032,49.817492,null,35.86166,29.31166,null,35.86166,28.0,null,20.593684,23.885942,null,35.86166,25.354826,null,15.870032,55.378051,null,-25.274398,-40.900557,null,35.86166,31.791702,null,46.603354,33.886917,null,30.375321,55.378051,null,35.86166,-38.416097,null,4.210484,37.09024,null,51.165691,47.162494,null,37.09024,-25.274398,null,51.165691,56.26392,null,47.162494,41.87194,null,-32.522779,-14.235004,null,46.603354,48.669026,null,51.919438,61.92411,null,51.919438,50.850346,null,56.130366,51.919438,null,51.919438,45.943161,null,39.399872,46.603354,null,52.132633,53.41291,null,46.603354,46.818188,null,20.593684,9.081999,null,35.86166,38.963745,null,31.046051,37.09024,null,49.817492,48.669026,null,35.907757,36.204824,null,35.86166,13.794185,null,38.963745,26.3351,null,52.132633,55.378051,null,45.943161,38.963745,null,12.565679,51.165691,null,38.963745,47.162494,null,31.046051,52.132633,null,35.86166,26.3351,null,35.907757,14.058324,null,55.378051,53.41291,null,14.058324,63.397768,null,37.09024,55.378051,null,56.879635,63.397768,null,46.603354,52.132633,null,47.411631,48.669026,null,48.669026,47.516231,null,46.603354,45.943161,null,51.919438,47.516231,null,55.378051,52.132633,null,26.820553,-19.015438,null,51.919438,63.397768,null,40.463667,51.165691,null,49.817492,46.603354,null,51.165691,33.886917,null,38.963745,39.074208,null,37.09024,35.86166,null,14.058324,35.86166,null,52.132633,49.817492,null,51.919438,41.87194,null,15.870032,-25.274398,null,20.593684,51.165691,null,14.058324,56.130366,null,15.870032,46.603354,null,47.162494,49.817492,null,45.943161,41.87194,null,37.09024,36.204824,null,51.165691,45.943161,null,38.963745,42.315407,null,51.919438,55.169438,null,35.86166,8.537981,null,20.593684,-0.023559,null,40.069099,61.52401,null,4.210484,-25.274398,null,51.919438,56.879635,null,55.378051,51.165691,null,1.352083,12.565679,null,20.593684,52.132633,null,20.593684,8.619543,null,55.169438,63.397768,null,52.132633,36.204824,null,35.86166,42.733883,null,12.879721,36.204824,null,35.907757,35.86166,null,51.165691,46.151241,null,15.870032,36.204824,null,51.165691,53.41291,null,42.315407,47.162494,null,4.210484,1.352083,null,23.634501,8.537981,null,51.165691,37.09024,null,38.963745,63.397768,null,35.86166,45.1,null,37.09024,8.619543,null,39.399872,48.669026,null,47.162494,40.463667,null,20.593684,15.870032,null,38.963745,46.603354,null,4.210484,15.870032,null,15.870032,38.963745,null,35.907757,22.396428,null,20.593684,55.378051,null,31.046051,-32.522779,null,31.046051,35.126413,null,20.593684,-6.369028,null,51.919438,55.378051,null,55.378051,46.603354,null,38.963745,55.378051,null,46.603354,51.919438,null,40.463667,-18.665695,null,35.86166,21.916221,null,31.791702,46.603354,null,35.86166,21.512583,null,14.058324,60.472024,null,35.907757,-0.789275,null,30.375321,55.169438,null,51.165691,55.169438,null,39.399872,51.165691,null,35.907757,37.09024,null,-30.559482,-19.015438,null,14.058324,45.943161,null,38.963745,44.016521,null,20.593684,46.603354,null,42.315407,48.669026,null,20.593684,33.223191,null,52.132633,56.26392,null,31.791702,39.399872,null,51.165691,39.399872,null,41.87194,51.165691,null,42.315407,52.132633,null,23.634501,56.130366,null,12.879721,20.593684,null,45.943161,42.733883,null,35.86166,-11.202692,null,35.86166,15.783471,null,50.850346,51.919438,null,47.162494,48.669026,null,52.132633,38.963745,null,46.603354,37.09024,null,-25.274398,31.046051,null,38.963745,42.733883,null,12.565679,48.669026,null,52.132633,23.424076,null,56.26392,60.472024,null,51.165691,31.791702,null,52.132633,46.818188,null,51.919438,46.818188,null,50.850346,40.463667,null,49.817492,47.162494,null,51.919438,40.463667,null,51.165691,38.963745,null,1.352083,7.873054,null,37.09024,51.165691,null,52.132633,63.397768,null,45.943161,39.074208,null,51.165691,61.92411,null,14.058324,4.210484,null,35.86166,33.854721,null,45.1,51.165691,null,46.603354,55.378051,null,41.87194,40.463667,null,35.86166,-1.831239,null,20.593684,-35.675147,null,38.963745,40.143105,null,46.603354,28.0,null,20.593684,21.512583,null,48.669026,46.603354,null,35.86166,15.552727,null,51.165691,28.0,null,20.593684,4.210484,null,37.09024,15.783471,null,14.058324,12.565679,null,15.870032,52.132633,null,35.86166,30.375321,null,22.396428,35.907757,null,44.016521,45.943161,null,35.86166,53.709807,null,51.919438,60.472024,null,37.09024,9.945587,null,51.165691,49.815273,null,52.132633,47.516231,null,15.870032,14.058324,null,38.963745,26.820553,null,49.817492,41.87194,null,38.963745,51.919438,null,35.907757,46.862496,null,35.86166,33.886917,null,52.132633,49.815273,null,56.879635,51.165691,null,39.399872,55.378051,null,48.669026,51.165691,null,20.593684,-30.559482,null,22.396428,36.204824,null,41.87194,46.818188,null,14.058324,-0.789275,null,36.204824,1.352083,null,49.817492,40.463667,null,38.963745,31.791702,null,1.352083,-25.274398,null,55.378051,-30.559482,null,37.09024,52.132633,null,51.165691,23.634501,null,51.165691,48.669026,null,46.603354,31.791702,null,20.593684,1.352083,null,63.397768,51.165691,null,35.907757,41.377491,null,52.132633,48.669026,null,35.86166,32.427908,null,46.151241,48.379433,null,35.86166,7.873054,null,42.315407,49.817492,null,4.210484,22.396428,null,14.058324,23.424076,null,40.463667,7.946527,null,35.86166,-6.369028,null,47.162494,44.016521,null,12.565679,40.463667,null,30.375321,50.850346,null,47.162494,45.1,null,51.165691,48.379433,null,49.817492,50.850346,null,35.86166,46.862496,null,48.669026,51.919438,null,46.603354,26.3351,null,35.86166,40.339852,null,12.879721,35.86166,null,31.046051,-0.228021,null,35.86166,44.016521,null,31.791702,51.165691,null,46.603354,39.399872,null,4.570868,23.634501,null,35.86166,5.152149,null,20.593684,-25.274398,null,35.86166,14.497401,null,50.850346,-30.559482,null,38.963745,28.0,null,35.86166,41.377491,null,35.86166,12.862807,null,58.595272,61.92411,null,51.165691,35.86166,null,22.396428,1.352083,null,12.565679,46.603354,null,40.463667,-30.559482,null,20.593684,56.130366,null,51.165691,60.472024,null,41.87194,37.09024,null,38.963745,49.817492,null,52.132633,55.169438,null,-30.559482,-22.95764,null,37.09024,23.424076,null,12.565679,56.130366,null,1.352083,4.210484,null,50.850346,28.0,null,51.165691,45.1,null,56.879635,55.169438,null,55.169438,51.165691,null,20.593684,51.919438,null,20.593684,30.375321,null,35.86166,11.825138,null,56.26392,63.397768,null,38.963745,31.046051,null,63.397768,55.378051,null,4.210484,36.204824,null,19.85627,56.130366,null,49.817492,46.818188,null,38.963745,61.92411,null,55.169438,51.919438,null,20.593684,41.87194,null,51.165691,44.016521,null,40.463667,55.378051,null,35.907757,32.427908,null,37.09024,7.51498,null,30.375321,41.87194,null,20.593684,29.31166,null,20.593684,40.463667,null,35.86166,41.0,null,38.963745,33.886917,null,53.41291,55.378051,null,35.86166,22.198745,null,52.132633,45.943161,null,40.463667,50.850346,null,37.09024,1.352083,null,30.375321,-25.274398,null,63.397768,46.603354,null,37.09024,12.879721,null,1.352083,-0.789275,null,49.817492,47.516231,null,36.204824,37.09024,null,14.058324,20.593684,null,41.87194,45.943161,null,38.963745,-32.522779,null,63.397768,56.879635,null,58.595272,63.397768,null,22.396428,51.165691,null,15.870032,21.00789,null,52.132633,56.130366,null,55.169438,56.26392,null,51.165691,47.411631,null,20.593684,23.685,null,1.352083,35.86166,null,12.565679,-40.900557,null,37.09024,-14.235004,null,46.151241,47.516231,null,-25.274398,37.09024,null,38.963745,23.885942,null,42.733883,45.943161,null,38.963745,50.850346,null,35.86166,28.394857,null,-23.442503,-35.675147,null,51.919438,42.733883,null,35.86166,7.369722,null,56.130366,-20.904305,null,42.315407,50.850346,null,61.92411,58.595272,null,19.85627,46.603354,null,14.058324,-25.274398,null,13.794185,15.199999,null,47.162494,47.516231,null,35.907757,38.963745,null,56.879635,58.595272,null,38.963745,23.424076,null,46.603354,49.817492,null,40.463667,49.817492,null,48.669026,49.817492,null,41.87194,38.963745,null,35.86166,56.879635,null,45.943161,50.850346,null,35.86166,-20.348404,null,36.204824,50.850346,null,4.210484,12.879721,null,39.399872,51.919438,null,36.204824,22.396428,null,51.919438,53.709807,null,42.733883,39.074208,null,30.375321,33.223191,null,39.399872,56.26392,null,35.86166,-18.665695,null,35.86166,58.595272,null,35.907757,1.352083,null,12.879721,37.09024,null,22.396428,14.058324,null,39.074208,31.046051,null,63.397768,37.09024,null,35.907757,10.691803,null,40.463667,39.074208,null,38.963745,4.210484,null,20.593684,56.26392,null,46.603354,63.397768,null,51.165691,26.820553,null,61.92411,63.397768,null,46.603354,49.815273,null,39.399872,-11.202692,null,36.204824,61.52401,null,47.411631,51.165691,null,37.09024,-30.559482,null,45.943161,48.669026,null,12.879721,-32.522779,null,45.1,44.016521,null,44.016521,49.817492,null,30.375321,40.463667,null,40.463667,51.919438,null,44.016521,51.165691,null,55.378051,38.963745,null,14.058324,15.870032,null,-0.789275,36.204824,null,-40.900557,-25.274398,null,49.817492,45.943161,null,35.86166,38.969719,null,-25.274398,35.907757,null,63.397768,64.963051,null,35.86166,34.802075,null,55.378051,-25.274398,null,48.379433,39.399872,null,52.132633,-14.235004,null,35.86166,21.521757,null,1.352083,36.204824,null,50.850346,63.397768,null,41.87194,41.0,null,35.86166,7.539989,null,51.919438,46.151241,null,55.378051,37.09024,null,50.850346,55.378051,null,14.058324,50.850346,null,30.375321,63.397768,null,38.963745,53.709807,null,20.593684,36.204824,null,35.86166,15.199999,null,51.919438,34.802075,null,35.86166,-2.1646,null,45.943161,45.1,null,39.399872,18.735693,null,63.397768,36.204824,null,48.669026,47.162494,null,38.963745,43.915886,null,51.165691,30.585164,null,37.09024,10.691803,null,45.943161,44.016521,null,55.378051,40.463667,null,39.399872,50.850346,null,35.86166,1.373333,null,14.058324,47.516231,null,43.915886,51.165691,null,23.634501,9.748917,null,-0.789275,1.352083,null,51.165691,42.733883,null,50.850346,49.817492,null,38.963745,30.585164,null,47.162494,33.886917,null,38.963745,-18.766947,null,45.943161,52.132633,null,51.919438,45.1,null,49.817492,55.378051,null,37.09024,53.41291,null,51.919438,58.595272,null,26.820553,23.424076,null,51.165691,43.915886,null,36.204824,35.907757,null,38.963745,35.126413,null,41.377491,61.52401,null,37.09024,46.818188,null,39.399872,63.397768,null,41.377491,48.019573,null,55.169438,52.132633,null,51.919438,37.09024,null,55.169438,58.595272,null,14.058324,12.879721,null,40.463667,-35.675147,null,56.130366,23.634501,null,41.87194,49.817492,null,52.132633,47.162494,null,41.87194,39.074208,null,38.963745,49.815273,null,40.463667,52.132633,null,-30.559482,-22.328474,null,58.595272,60.472024,null,12.565679,35.86166,null,51.165691,-30.559482,null,22.396428,-25.274398,null,46.603354,47.516231,null,21.916221,15.870032,null,42.315407,41.87194,null,42.315407,40.463667,null,30.375321,-38.416097,null,56.26392,37.09024,null,31.046051,50.850346,null],"line":{"color":"rgba(0, 150, 0, 0.5)","width":0.5013167323698858},"lon":[104.195397,10.451526,null,-102.552784,-95.712891,null,104.195397,-3.435973,null,104.195397,5.291266,null,104.195397,133.775136,null,104.195397,-106.346771,null,-95.712891,-102.552784,null,104.195397,127.766922,null,104.195397,1.888334,null,5.291266,10.451526,null,46.869107,1.888334,null,104.195397,-102.552784,null,-95.712891,-106.346771,null,104.195397,100.992541,null,104.195397,19.145136,null,78.96288,53.847818,null,104.195397,113.921327,null,5.291266,4.351721,null,104.195397,121.774017,null,104.195397,-3.74922,null,104.195397,45.079162,null,104.195397,4.351721,null,104.195397,78.96288,null,104.195397,108.277199,null,104.195397,101.975766,null,108.277199,-95.712891,null,104.195397,12.56738,null,10.451526,1.888334,null,19.145136,10.451526,null,104.195397,105.318756,null,10.451526,19.145136,null,35.243322,-95.712891,null,5.291266,1.888334,null,104.990963,-95.712891,null,104.195397,16.354896,null,-70.162651,-95.712891,null,104.195397,-51.92528,null,104.195397,53.847818,null,104.195397,-71.542969,null,104.195397,22.937506,null,78.96288,50.55096,null,10.451526,14.550072,null,100.992541,-95.712891,null,69.345116,-95.712891,null,-106.346771,-95.712891,null,10.451526,12.56738,null,78.96288,-95.712891,null,10.451526,15.472962,null,108.277199,10.451526,null,-3.435973,-7.09262,null,104.195397,103.819836,null,10.451526,-3.74922,null,104.195397,114.109497,null,10.451526,8.227512,null,-7.09262,-3.74922,null,104.195397,21.824312,null,-3.74922,-8.224454,null,108.277199,138.252924,null,104.195397,9.501785,null,19.145136,15.472962,null,108.277199,127.766922,null,43.356892,19.145136,null,46.869107,104.195397,null,35.243322,10.451526,null,104.195397,34.851612,null,35.243322,43.679291,null,104.195397,-74.297333,null,10.451526,5.291266,null,104.195397,66.923684,null,-3.74922,12.56738,null,104.195397,14.995463,null,100.992541,10.451526,null,15.472962,10.451526,null,101.975766,35.243322,null,95.955974,-95.712891,null,-3.74922,1.888334,null,104.195397,8.675277,null,104.195397,43.679291,null,16.354896,8.468946,null,-3.74922,-7.09262,null,104.195397,37.906193,null,104.195397,8.468946,null,10.451526,4.351721,null,15.472962,19.145136,null,104.195397,90.3563,null,104.195397,-75.015152,null,104.195397,174.885971,null,-8.224454,-3.74922,null,1.888334,12.56738,null,5.291266,-95.712891,null,35.243322,5.291266,null,19.145136,5.291266,null,24.96676,19.145136,null,5.291266,-3.74922,null,35.243322,28.369885,null,35.243322,12.56738,null,1.888334,10.451526,null,10.451526,-3.435973,null,10.451526,16.354896,null,4.351721,10.451526,null,104.195397,104.990963,null,138.252924,108.277199,null,104.195397,-8.224454,null,101.975766,1.888334,null,104.195397,8.227512,null,5.291266,12.56738,null,1.888334,-3.74922,null,12.56738,1.888334,null,95.955974,127.766922,null,104.195397,-8.24389,null,69.345116,10.451526,null,5.291266,19.145136,null,4.351721,5.291266,null,-8.224454,15.472962,null,19.145136,1.888334,null,35.243322,-3.74922,null,24.96676,10.451526,null,19.145136,19.503304,null,-58.443832,-51.92528,null,104.195397,15.472962,null,16.354896,19.145136,null,15.472962,-95.712891,null,19.145136,19.699024,null,104.195397,25.748151,null,1.888334,4.351721,null,104.195397,24.96676,null,104.195397,19.503304,null,108.277199,12.56738,null,35.243322,24.96676,null,-95.712891,-70.162651,null,108.277199,-3.74922,null,104.195397,74.766098,null,114.109497,104.195397,null,4.351721,1.888334,null,104.195397,30.802498,null,104.195397,14.550072,null,104.195397,-70.162651,null,104.195397,-83.753428,null,138.252924,104.195397,null,-8.224454,3.0,null,104.195397,23.881275,null,16.354896,9.501785,null,19.145136,31.16558,null,16.354896,25.748151,null,-7.09262,-3.435973,null,104.195397,-66.58973,null,104.195397,31.16558,null,104.195397,19.699024,null,28.369885,24.96676,null,108.277199,-3.435973,null,104.195397,-55.765835,null,78.96288,-14.452362,null,78.96288,80.771797,null,69.345116,5.291266,null,108.277199,1.888334,null,-95.712891,4.351721,null,104.195397,36.238414,null,108.277199,19.145136,null,19.145136,9.501785,null,108.277199,5.291266,null,25.48583,10.451526,null,104.990963,138.252924,null,24.96676,1.888334,null,104.990963,133.775136,null,104.195397,-1.023194,null,64.585262,71.276093,null,114.109497,-95.712891,null,19.503304,10.451526,null,100.992541,15.472962,null,104.195397,47.481766,null,104.195397,3.0,null,78.96288,45.079162,null,104.195397,51.183884,null,100.992541,-3.435973,null,133.775136,174.885971,null,104.195397,-7.09262,null,1.888334,9.537499,null,69.345116,-3.435973,null,104.195397,-63.616672,null,101.975766,-95.712891,null,10.451526,19.503304,null,-95.712891,133.775136,null,10.451526,9.501785,null,19.503304,12.56738,null,-55.765835,-51.92528,null,1.888334,19.699024,null,19.145136,25.748151,null,19.145136,4.351721,null,-106.346771,19.145136,null,19.145136,24.96676,null,-8.224454,1.888334,null,5.291266,-8.24389,null,1.888334,8.227512,null,78.96288,8.675277,null,104.195397,35.243322,null,34.851612,-95.712891,null,15.472962,19.699024,null,127.766922,138.252924,null,104.195397,-88.89653,null,35.243322,17.228331,null,5.291266,-3.435973,null,24.96676,35.243322,null,104.990963,10.451526,null,35.243322,19.503304,null,34.851612,5.291266,null,104.195397,17.228331,null,127.766922,108.277199,null,-3.435973,-8.24389,null,108.277199,16.354896,null,-95.712891,-3.435973,null,24.603189,16.354896,null,1.888334,5.291266,null,28.369885,19.699024,null,19.699024,14.550072,null,1.888334,24.96676,null,19.145136,14.550072,null,-3.435973,5.291266,null,30.802498,29.154857,null,19.145136,16.354896,null,-3.74922,10.451526,null,15.472962,1.888334,null,10.451526,9.537499,null,35.243322,21.824312,null,-95.712891,104.195397,null,108.277199,104.195397,null,5.291266,15.472962,null,19.145136,12.56738,null,100.992541,133.775136,null,78.96288,10.451526,null,108.277199,-106.346771,null,100.992541,1.888334,null,19.503304,15.472962,null,24.96676,12.56738,null,-95.712891,138.252924,null,10.451526,24.96676,null,35.243322,43.356892,null,19.145136,23.881275,null,104.195397,-80.782127,null,78.96288,37.906193,null,45.038189,105.318756,null,101.975766,133.775136,null,19.145136,24.603189,null,-3.435973,10.451526,null,103.819836,104.990963,null,78.96288,5.291266,null,78.96288,0.824782,null,23.881275,16.354896,null,5.291266,138.252924,null,104.195397,25.48583,null,121.774017,138.252924,null,127.766922,104.195397,null,10.451526,14.995463,null,100.992541,138.252924,null,10.451526,-8.24389,null,43.356892,19.503304,null,101.975766,103.819836,null,-102.552784,-80.782127,null,10.451526,-95.712891,null,35.243322,16.354896,null,104.195397,15.2,null,-95.712891,0.824782,null,-8.224454,19.699024,null,19.503304,-3.74922,null,78.96288,100.992541,null,35.243322,1.888334,null,101.975766,100.992541,null,100.992541,35.243322,null,127.766922,114.109497,null,78.96288,-3.435973,null,34.851612,-55.765835,null,34.851612,33.429859,null,78.96288,34.888822,null,19.145136,-3.435973,null,-3.435973,1.888334,null,35.243322,-3.435973,null,1.888334,19.145136,null,-3.74922,35.529562,null,104.195397,95.955974,null,-7.09262,1.888334,null,104.195397,55.923255,null,108.277199,8.468946,null,127.766922,113.921327,null,69.345116,23.881275,null,10.451526,23.881275,null,-8.224454,10.451526,null,127.766922,-95.712891,null,22.937506,29.154857,null,108.277199,24.96676,null,35.243322,21.005859,null,78.96288,1.888334,null,43.356892,19.699024,null,78.96288,43.679291,null,5.291266,9.501785,null,-7.09262,-8.224454,null,10.451526,-8.224454,null,12.56738,10.451526,null,43.356892,5.291266,null,-102.552784,-106.346771,null,121.774017,78.96288,null,24.96676,25.48583,null,104.195397,17.873887,null,104.195397,-90.230759,null,4.351721,19.145136,null,19.503304,19.699024,null,5.291266,35.243322,null,1.888334,-95.712891,null,133.775136,34.851612,null,35.243322,25.48583,null,104.990963,19.699024,null,5.291266,53.847818,null,9.501785,8.468946,null,10.451526,-7.09262,null,5.291266,8.227512,null,19.145136,8.227512,null,4.351721,-3.74922,null,15.472962,19.503304,null,19.145136,-3.74922,null,10.451526,35.243322,null,103.819836,80.771797,null,-95.712891,10.451526,null,5.291266,16.354896,null,24.96676,21.824312,null,10.451526,25.748151,null,108.277199,101.975766,null,104.195397,35.862285,null,15.2,10.451526,null,1.888334,-3.435973,null,12.56738,-3.74922,null,104.195397,-78.183406,null,78.96288,-71.542969,null,35.243322,47.576927,null,1.888334,3.0,null,78.96288,55.923255,null,19.699024,1.888334,null,104.195397,48.516388,null,10.451526,3.0,null,78.96288,101.975766,null,-95.712891,-90.230759,null,108.277199,104.990963,null,100.992541,5.291266,null,104.195397,69.345116,null,114.109497,127.766922,null,21.005859,24.96676,null,104.195397,27.953389,null,19.145136,8.468946,null,-95.712891,-9.696645,null,10.451526,6.129583,null,5.291266,14.550072,null,100.992541,108.277199,null,35.243322,30.802498,null,15.472962,12.56738,null,35.243322,19.145136,null,127.766922,103.846656,null,104.195397,9.537499,null,5.291266,6.129583,null,24.603189,10.451526,null,-8.224454,-3.435973,null,19.699024,10.451526,null,78.96288,22.937506,null,114.109497,138.252924,null,12.56738,8.227512,null,108.277199,113.921327,null,138.252924,103.819836,null,15.472962,-3.74922,null,35.243322,-7.09262,null,103.819836,133.775136,null,-3.435973,22.937506,null,-95.712891,5.291266,null,10.451526,-102.552784,null,10.451526,19.699024,null,1.888334,-7.09262,null,78.96288,103.819836,null,16.354896,10.451526,null,127.766922,64.585262,null,5.291266,19.699024,null,104.195397,53.688046,null,14.995463,31.16558,null,104.195397,80.771797,null,43.356892,15.472962,null,101.975766,114.109497,null,108.277199,53.847818,null,-3.74922,-1.023194,null,104.195397,34.888822,null,19.503304,21.005859,null,104.990963,-3.74922,null,69.345116,4.351721,null,19.503304,15.2,null,10.451526,31.16558,null,15.472962,4.351721,null,104.195397,103.846656,null,19.699024,19.145136,null,1.888334,17.228331,null,104.195397,127.510093,null,121.774017,104.195397,null,34.851612,15.827659,null,104.195397,21.005859,null,-7.09262,10.451526,null,1.888334,-8.224454,null,-74.297333,-102.552784,null,104.195397,46.199616,null,78.96288,133.775136,null,104.195397,-14.452362,null,4.351721,22.937506,null,35.243322,3.0,null,104.195397,64.585262,null,104.195397,30.217636,null,25.013607,25.748151,null,10.451526,104.195397,null,114.109497,103.819836,null,104.990963,1.888334,null,-3.74922,22.937506,null,78.96288,-106.346771,null,10.451526,8.468946,null,12.56738,-95.712891,null,35.243322,15.472962,null,5.291266,23.881275,null,22.937506,18.49041,null,-95.712891,53.847818,null,104.990963,-106.346771,null,103.819836,101.975766,null,4.351721,3.0,null,10.451526,15.2,null,24.603189,23.881275,null,23.881275,10.451526,null,78.96288,19.145136,null,78.96288,69.345116,null,104.195397,42.590275,null,9.501785,16.354896,null,35.243322,34.851612,null,16.354896,-3.435973,null,101.975766,138.252924,null,102.495496,-106.346771,null,15.472962,8.227512,null,35.243322,25.748151,null,23.881275,19.145136,null,78.96288,12.56738,null,10.451526,21.005859,null,-3.74922,-3.435973,null,127.766922,53.688046,null,-95.712891,134.58252,null,69.345116,12.56738,null,78.96288,47.481766,null,78.96288,-3.74922,null,104.195397,20.0,null,35.243322,9.537499,null,-8.24389,-3.435973,null,104.195397,113.543873,null,5.291266,24.96676,null,-3.74922,4.351721,null,-95.712891,103.819836,null,69.345116,133.775136,null,16.354896,1.888334,null,-95.712891,121.774017,null,103.819836,113.921327,null,15.472962,14.550072,null,138.252924,-95.712891,null,108.277199,78.96288,null,12.56738,24.96676,null,35.243322,-55.765835,null,16.354896,24.603189,null,25.013607,16.354896,null,114.109497,10.451526,null,100.992541,-10.940835,null,5.291266,-106.346771,null,23.881275,9.501785,null,10.451526,28.369885,null,78.96288,90.3563,null,103.819836,104.195397,null,104.990963,174.885971,null,-95.712891,-51.92528,null,14.995463,14.550072,null,133.775136,-95.712891,null,35.243322,45.079162,null,25.48583,24.96676,null,35.243322,4.351721,null,104.195397,84.124008,null,-58.443832,-71.542969,null,19.145136,25.48583,null,104.195397,12.354722,null,-106.346771,165.618042,null,43.356892,4.351721,null,25.748151,25.013607,null,102.495496,1.888334,null,108.277199,133.775136,null,-88.89653,-86.241905,null,19.503304,14.550072,null,127.766922,35.243322,null,24.603189,25.013607,null,35.243322,53.847818,null,1.888334,15.472962,null,-3.74922,15.472962,null,19.699024,15.472962,null,12.56738,35.243322,null,104.195397,24.603189,null,24.96676,4.351721,null,104.195397,57.552152,null,138.252924,4.351721,null,101.975766,121.774017,null,-8.224454,19.145136,null,138.252924,114.109497,null,19.145136,27.953389,null,25.48583,21.824312,null,69.345116,43.679291,null,-8.224454,9.501785,null,104.195397,35.529562,null,104.195397,25.013607,null,127.766922,103.819836,null,121.774017,-95.712891,null,114.109497,108.277199,null,21.824312,34.851612,null,16.354896,-95.712891,null,127.766922,-61.222503,null,-3.74922,21.824312,null,35.243322,101.975766,null,78.96288,9.501785,null,1.888334,16.354896,null,10.451526,30.802498,null,25.748151,16.354896,null,1.888334,6.129583,null,-8.224454,17.873887,null,138.252924,105.318756,null,28.369885,10.451526,null,-95.712891,22.937506,null,24.96676,19.699024,null,121.774017,-55.765835,null,15.2,21.005859,null,21.005859,15.472962,null,69.345116,-3.74922,null,-3.74922,19.145136,null,21.005859,10.451526,null,-3.435973,35.243322,null,108.277199,100.992541,null,113.921327,138.252924,null,174.885971,133.775136,null,15.472962,24.96676,null,104.195397,59.556278,null,133.775136,127.766922,null,16.354896,-19.020835,null,104.195397,38.996815,null,-3.435973,133.775136,null,31.16558,-8.224454,null,5.291266,-51.92528,null,104.195397,-77.781167,null,103.819836,138.252924,null,4.351721,16.354896,null,12.56738,20.0,null,104.195397,-5.54708,null,19.145136,14.995463,null,-3.435973,-95.712891,null,4.351721,-3.435973,null,108.277199,4.351721,null,69.345116,16.354896,null,35.243322,27.953389,null,78.96288,138.252924,null,104.195397,-86.241905,null,19.145136,38.996815,null,104.195397,24.15536,null,24.96676,15.2,null,-8.224454,-70.162651,null,16.354896,138.252924,null,19.699024,19.503304,null,35.243322,17.679076,null,10.451526,36.238414,null,-95.712891,-61.222503,null,24.96676,21.005859,null,-3.435973,-3.74922,null,-8.224454,4.351721,null,104.195397,32.290275,null,108.277199,14.550072,null,17.679076,10.451526,null,-102.552784,-83.753428,null,113.921327,103.819836,null,10.451526,25.48583,null,4.351721,15.472962,null,35.243322,36.238414,null,19.503304,9.537499,null,35.243322,46.869107,null,24.96676,5.291266,null,19.145136,15.2,null,15.472962,-3.435973,null,-95.712891,-8.24389,null,19.145136,25.013607,null,30.802498,53.847818,null,10.451526,17.679076,null,138.252924,127.766922,null,35.243322,33.429859,null,64.585262,105.318756,null,-95.712891,8.227512,null,-8.224454,16.354896,null,64.585262,66.923684,null,23.881275,5.291266,null,19.145136,-95.712891,null,23.881275,25.013607,null,108.277199,121.774017,null,-3.74922,-71.542969,null,-106.346771,-102.552784,null,12.56738,15.472962,null,5.291266,19.503304,null,12.56738,21.824312,null,35.243322,6.129583,null,-3.74922,5.291266,null,22.937506,24.684866,null,25.013607,8.468946,null,104.990963,104.195397,null,10.451526,22.937506,null,114.109497,133.775136,null,1.888334,14.550072,null,95.955974,100.992541,null,43.356892,12.56738,null,43.356892,-3.74922,null,69.345116,-63.616672,null,9.501785,-95.712891,null,34.851612,4.351721,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[-11.202692,21.521757,0.18636,-30.559482,17.060816,15.414999,40.143105,42.315407,61.52401,55.378051,-38.416097,-35.675147,-9.189967,-32.522779,-25.274398,50.850346,-22.328474,-14.235004,-9.64571,56.130366,35.86166,-10.447525,56.26392,-17.713371,46.603354,7.946527,22.396428,-0.789275,41.87194,7.539989,36.204824,35.907757,4.210484,46.862496,-0.522778,-40.900557,60.472024,7.51498,8.537981,-6.314993,12.879721,-8.874217,14.497401,1.352083,40.463667,46.818188,15.870032,23.424076,38.963745,26.820553,37.09024,12.238333,26.02751,29.31166,25.354826,23.885942,40.069099,53.709807,39.074208,12.16957,13.193887,12.262776,12.984305,10.691803,28.0,42.546245,47.516231,23.685,43.915886,42.733883,7.369722,4.570868,-0.228021,-2.1646,9.748917,45.1,35.126413,49.817492,9.145,58.595272,61.892635,61.92411,-0.803689,13.443182,51.165691,36.140751,18.971187,47.162494,64.963051,33.223191,53.41291,31.046051,48.019573,30.585164,-0.023559,33.854721,56.879635,55.169438,49.815273,35.937496,23.634501,47.411631,42.708678,31.791702,21.512583,52.132633,-20.904305,17.607789,9.081999,30.375321,51.919438,39.399872,45.943161,43.94236,44.016521,20.593684,48.669026,14.058324,46.151241,63.397768,8.619543,33.886917,48.379433,41.608635,-6.369028,6.42375,-13.133897,-16.290154,25.03428,18.735693,-1.831239,13.794185,15.783471,15.199999,18.109581,6.428055,7.131474,-23.442503,17.189877,4.535277,41.0,32.427908,21.916221,12.565679,4.860416,41.20438,46.946947,21.694025,32.3078,-3.373056,16.5388,19.3133,7.873054,15.454166,-11.6455,9.30769,1.650801,-17.679742,11.825138,31.952162,71.706936,9.945587,40.339852,19.85627,-29.609988,26.3351,22.198745,-18.766947,-13.254308,3.202778,17.570692,21.00789,-20.348404,-18.665695,-22.95764,28.394857,12.52111,-15.376706,12.865416,-1.940278,17.357822,8.460555,5.152149,-19.015438,12.862807,3.919305,-26.522503,38.861034,-21.178986,38.969719,1.373333,41.377491,-13.759029,15.552727,-3.370417,-7.109535,-13.768752,6.611111,-49.280366,18.04248,11.803749,-4.679574,13.444304,7.425554,12.20189,-21.236736,-19.054445,33.0,27.514162,15.179384,41.902916,13.909444,34.802075,-51.796253,-15.965,18.420695,18.218785,10.961632,17.897476,7.862684,-6.343194,-14.28522,16.742498,17.664332,-29.040835,-90.0,-24.376515],"lon":[17.873887,-77.781167,6.613081,22.937506,-61.796428,-61.370976,47.576927,43.356892,105.318756,-3.435973,-63.616672,-71.542969,-75.015152,-55.765835,133.775136,4.351721,24.684866,-51.92528,160.156194,-106.346771,104.195397,105.690449,9.501785,178.065033,1.888334,-1.023194,114.109497,113.921327,12.56738,-5.54708,138.252924,127.766922,101.975766,103.846656,166.931503,174.885971,8.468946,134.58252,-80.782127,143.95555,121.774017,125.727,-14.452362,103.819836,-3.74922,8.227512,100.992541,53.847818,35.243322,30.802498,-95.712891,-1.561593,50.55096,47.481766,51.183884,45.079162,45.038189,27.953389,21.824312,-68.990021,-59.543198,-61.604171,-61.287228,-61.222503,3.0,1.601554,14.550072,90.3563,17.679076,25.48583,12.354722,-74.297333,15.827659,24.15536,-83.753428,15.2,33.429859,15.472962,40.489673,25.013607,-6.911806,25.748151,11.609444,-15.310139,10.451526,-5.353585,-72.285215,19.503304,-19.020835,43.679291,-8.24389,34.851612,66.923684,36.238414,37.906193,35.862285,24.603189,23.881275,6.129583,14.375416,-102.552784,28.369885,19.37439,-7.09262,55.923255,5.291266,165.618042,8.081666,8.675277,69.345116,19.145136,-8.224454,24.96676,12.457777,21.005859,78.96288,19.699024,108.277199,14.995463,16.354896,0.824782,9.537499,31.16558,21.745275,34.888822,-66.58973,27.849332,-63.588653,-77.39628,-70.162651,-78.183406,-88.89653,-90.230759,-86.241905,-77.297508,-9.429499,171.184478,-58.443832,-88.49765,114.727669,20.0,53.688046,95.955974,104.990963,-58.93018,74.766098,-56.32509,-71.797928,-64.7505,29.918886,-23.0418,-81.2546,80.771797,18.732207,43.3333,2.315834,10.267895,-149.406843,42.590275,35.233154,-42.604303,-9.696645,127.510093,102.495496,28.233608,17.228331,113.543873,46.869107,34.301525,73.22068,-3.996166,-10.940835,57.552152,35.529562,18.49041,84.124008,-69.968338,166.959158,-85.207229,29.873888,-62.782998,-11.779889,46.199616,29.154857,30.217636,-56.027783,31.465866,71.276093,-175.198242,59.556278,32.290275,64.585262,-172.104629,48.516388,-168.734039,179.194167,-177.156097,20.939444,69.348557,-63.05483,-15.180413,55.491977,144.793731,150.550812,-68.262383,-159.777671,-169.867233,65.0,90.433601,39.782334,12.453389,-60.978893,38.996815,-59.523613,-5.7089,-64.639968,-63.043653,-169.09022,-62.83055,30.217636,71.876519,-170.70444,-62.187366,145.94351,167.954712,0.0,-128.324001],"marker":{"color":"red","opacity":0.8,"size":[20.0,8.0,2.4,20.0,10.0,4.4,20.0,20.0,18.4,20.0,18.8,20.0,16.0,20.0,20.0,20.0,10.0,20.0,3.6,20.0,20.0,0.8,20.0,14.4,20.0,19.6,20.0,20.0,20.0,12.4,20.0,20.0,20.0,10.8,2.4,20.0,20.0,2.0,18.8,6.0,20.0,4.8,12.4,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,14.4,20.0,20.0,20.0,20.0,17.6,8.8,20.0,8.0,15.2,6.4,4.4,20.0,10.0,7.6,20.0,10.8,20.0,20.0,8.8,20.0,9.2,10.4,16.0,20.0,20.0,20.0,10.8,20.0,6.0,20.0,7.6,6.0,20.0,6.8,5.6,20.0,16.4,11.2,20.0,20.0,20.0,15.2,20.0,14.0,20.0,20.0,20.0,20.0,20.0,18.8,14.0,20.0,20.0,20.0,10.0,6.4,12.4,20.0,20.0,20.0,20.0,3.2,20.0,20.0,20.0,20.0,20.0,20.0,6.4,11.6,20.0,20.0,15.6,8.4,8.8,10.4,8.4,20.0,11.6,16.0,20.0,9.2,8.8,8.4,4.4,14.4,6.0,14.0,13.6,6.4,15.2,20.0,14.4,16.8,0.8,4.0,6.4,2.8,4.8,6.0,11.2,4.4,2.4,5.6,6.0,7.6,6.0,4.0,5.2,7.6,0.4,9.2,1.2,10.0,8.8,16.8,4.8,10.0,8.4,7.6,20.0,10.8,20.0,6.0,6.0,4.4,13.6,6.0,4.4,5.6,6.0,6.4,6.8,7.2,3.6,5.2,1.6,4.8,8.0,18.4,2.0,6.0,2.0,1.2,2.4,4.0,1.2,4.0,3.2,9.2,4.8,3.6,1.6,2.8,1.6,5.6,3.6,1.6,0.4,4.8,2.8,1.2,0.4,4.8,2.0,0.4,1.6,4.4,0.4,2.4,0.8,1.6,0.4,1.2,0.4]},"mode":"markers","name":"2023","text":["AGO\u003cbr\u003eConex\u00f5es: 50","CUB\u003cbr\u003eConex\u00f5es: 20","STP\u003cbr\u003eConex\u00f5es: 6","ZAF\u003cbr\u003eConex\u00f5es: 165","ATG\u003cbr\u003eConex\u00f5es: 25","DMA\u003cbr\u003eConex\u00f5es: 11","AZE\u003cbr\u003eConex\u00f5es: 56","GEO\u003cbr\u003eConex\u00f5es: 67","RUS\u003cbr\u003eConex\u00f5es: 46","GBR\u003cbr\u003eConex\u00f5es: 231","ARG\u003cbr\u003eConex\u00f5es: 47","CHL\u003cbr\u003eConex\u00f5es: 66","PER\u003cbr\u003eConex\u00f5es: 40","URY\u003cbr\u003eConex\u00f5es: 52","AUS\u003cbr\u003eConex\u00f5es: 157","BEL\u003cbr\u003eConex\u00f5es: 226","BWA\u003cbr\u003eConex\u00f5es: 25","BRA\u003cbr\u003eConex\u00f5es: 152","SLB\u003cbr\u003eConex\u00f5es: 9","CAN\u003cbr\u003eConex\u00f5es: 180","CHN\u003cbr\u003eConex\u00f5es: 275","CXR\u003cbr\u003eConex\u00f5es: 2","DNK\u003cbr\u003eConex\u00f5es: 161","FJI\u003cbr\u003eConex\u00f5es: 36","FRA\u003cbr\u003eConex\u00f5es: 248","GHA\u003cbr\u003eConex\u00f5es: 49","HKG\u003cbr\u003eConex\u00f5es: 167","IDN\u003cbr\u003eConex\u00f5es: 122","ITA\u003cbr\u003eConex\u00f5es: 214","CIV\u003cbr\u003eConex\u00f5es: 31","JPN\u003cbr\u003eConex\u00f5es: 148","KOR\u003cbr\u003eConex\u00f5es: 189","MYS\u003cbr\u003eConex\u00f5es: 141","MNG\u003cbr\u003eConex\u00f5es: 27","NRU\u003cbr\u003eConex\u00f5es: 6","NZL\u003cbr\u003eConex\u00f5es: 114","NOR\u003cbr\u003eConex\u00f5es: 135","PLW\u003cbr\u003eConex\u00f5es: 5","PAN\u003cbr\u003eConex\u00f5es: 47","PNG\u003cbr\u003eConex\u00f5es: 15","PHL\u003cbr\u003eConex\u00f5es: 82","TLS\u003cbr\u003eConex\u00f5es: 12","SEN\u003cbr\u003eConex\u00f5es: 31","SGP\u003cbr\u003eConex\u00f5es: 148","ESP\u003cbr\u003eConex\u00f5es: 208","CHE\u003cbr\u003eConex\u00f5es: 204","THA\u003cbr\u003eConex\u00f5es: 156","ARE\u003cbr\u003eConex\u00f5es: 74","TUR\u003cbr\u003eConex\u00f5es: 238","EGY\u003cbr\u003eConex\u00f5es: 62","USA\u003cbr\u003eConex\u00f5es: 254","BFA\u003cbr\u003eConex\u00f5es: 36","BHR\u003cbr\u003eConex\u00f5es: 56","KWT\u003cbr\u003eConex\u00f5es: 54","QAT\u003cbr\u003eConex\u00f5es: 50","SAU\u003cbr\u003eConex\u00f5es: 67","ARM\u003cbr\u003eConex\u00f5es: 44","BLR\u003cbr\u003eConex\u00f5es: 22","GRC\u003cbr\u003eConex\u00f5es: 129","CUW\u003cbr\u003eConex\u00f5es: 20","BRB\u003cbr\u003eConex\u00f5es: 38","GRD\u003cbr\u003eConex\u00f5es: 16","VCT\u003cbr\u003eConex\u00f5es: 11","TTO\u003cbr\u003eConex\u00f5es: 60","DZA\u003cbr\u003eConex\u00f5es: 25","AND\u003cbr\u003eConex\u00f5es: 19","AUT\u003cbr\u003eConex\u00f5es: 52","BGD\u003cbr\u003eConex\u00f5es: 27","BIH\u003cbr\u003eConex\u00f5es: 58","BGR\u003cbr\u003eConex\u00f5es: 115","CMR\u003cbr\u003eConex\u00f5es: 22","COL\u003cbr\u003eConex\u00f5es: 89","COG\u003cbr\u003eConex\u00f5es: 23","COD\u003cbr\u003eConex\u00f5es: 26","CRI\u003cbr\u003eConex\u00f5es: 40","HRV\u003cbr\u003eConex\u00f5es: 100","CYP\u003cbr\u003eConex\u00f5es: 60","CZE\u003cbr\u003eConex\u00f5es: 171","ETH\u003cbr\u003eConex\u00f5es: 27","EST\u003cbr\u003eConex\u00f5es: 107","FRO\u003cbr\u003eConex\u00f5es: 15","FIN\u003cbr\u003eConex\u00f5es: 140","GAB\u003cbr\u003eConex\u00f5es: 19","GMB\u003cbr\u003eConex\u00f5es: 15","DEU\u003cbr\u003eConex\u00f5es: 255","GIB\u003cbr\u003eConex\u00f5es: 17","HTI\u003cbr\u003eConex\u00f5es: 14","HUN\u003cbr\u003eConex\u00f5es: 124","ISL\u003cbr\u003eConex\u00f5es: 41","IRQ\u003cbr\u003eConex\u00f5es: 28","IRL\u003cbr\u003eConex\u00f5es: 105","ISR\u003cbr\u003eConex\u00f5es: 95","KAZ\u003cbr\u003eConex\u00f5es: 65","JOR\u003cbr\u003eConex\u00f5es: 38","KEN\u003cbr\u003eConex\u00f5es: 83","LBN\u003cbr\u003eConex\u00f5es: 35","LVA\u003cbr\u003eConex\u00f5es: 94","LTU\u003cbr\u003eConex\u00f5es: 122","LUX\u003cbr\u003eConex\u00f5es: 92","MLT\u003cbr\u003eConex\u00f5es: 63","MEX\u003cbr\u003eConex\u00f5es: 79","MDA\u003cbr\u003eConex\u00f5es: 47","MNE\u003cbr\u003eConex\u00f5es: 35","MAR\u003cbr\u003eConex\u00f5es: 95","OMN\u003cbr\u003eConex\u00f5es: 64","NLD\u003cbr\u003eConex\u00f5es: 239","NCL\u003cbr\u003eConex\u00f5es: 25","NER\u003cbr\u003eConex\u00f5es: 16","NGA\u003cbr\u003eConex\u00f5es: 31","PAK\u003cbr\u003eConex\u00f5es: 129","POL\u003cbr\u003eConex\u00f5es: 191","PRT\u003cbr\u003eConex\u00f5es: 174","ROU\u003cbr\u003eConex\u00f5es: 120","SMR\u003cbr\u003eConex\u00f5es: 8","SRB\u003cbr\u003eConex\u00f5es: 82","IND\u003cbr\u003eConex\u00f5es: 224","SVK\u003cbr\u003eConex\u00f5es: 127","VNM\u003cbr\u003eConex\u00f5es: 108","SVN\u003cbr\u003eConex\u00f5es: 123","SWE\u003cbr\u003eConex\u00f5es: 179","TGO\u003cbr\u003eConex\u00f5es: 16","TUN\u003cbr\u003eConex\u00f5es: 29","UKR\u003cbr\u003eConex\u00f5es: 95","MKD\u003cbr\u003eConex\u00f5es: 51","TZA\u003cbr\u003eConex\u00f5es: 39","VEN\u003cbr\u003eConex\u00f5es: 21","ZMB\u003cbr\u003eConex\u00f5es: 22","BOL\u003cbr\u003eConex\u00f5es: 26","BHS\u003cbr\u003eConex\u00f5es: 21","DOM\u003cbr\u003eConex\u00f5es: 58","ECU\u003cbr\u003eConex\u00f5es: 29","SLV\u003cbr\u003eConex\u00f5es: 40","GTM\u003cbr\u003eConex\u00f5es: 50","HND\u003cbr\u003eConex\u00f5es: 23","JAM\u003cbr\u003eConex\u00f5es: 22","LBR\u003cbr\u003eConex\u00f5es: 21","MHL\u003cbr\u003eConex\u00f5es: 11","PRY\u003cbr\u003eConex\u00f5es: 36","BLZ\u003cbr\u003eConex\u00f5es: 15","BRN\u003cbr\u003eConex\u00f5es: 35","ALB\u003cbr\u003eConex\u00f5es: 34","IRN\u003cbr\u003eConex\u00f5es: 16","MMR\u003cbr\u003eConex\u00f5es: 38","KHM\u003cbr\u003eConex\u00f5es: 62","GUY\u003cbr\u003eConex\u00f5es: 36","KGZ\u003cbr\u003eConex\u00f5es: 42","SPM\u003cbr\u003eConex\u00f5es: 2","TCA\u003cbr\u003eConex\u00f5es: 10","BMU\u003cbr\u003eConex\u00f5es: 16","BDI\u003cbr\u003eConex\u00f5es: 7","CPV\u003cbr\u003eConex\u00f5es: 12","CYM\u003cbr\u003eConex\u00f5es: 15","LKA\u003cbr\u003eConex\u00f5es: 28","TCD\u003cbr\u003eConex\u00f5es: 11","COM\u003cbr\u003eConex\u00f5es: 6","BEN\u003cbr\u003eConex\u00f5es: 14","GNQ\u003cbr\u003eConex\u00f5es: 15","PYF\u003cbr\u003eConex\u00f5es: 19","DJI\u003cbr\u003eConex\u00f5es: 15","PSE\u003cbr\u003eConex\u00f5es: 10","GRL\u003cbr\u003eConex\u00f5es: 13","GIN\u003cbr\u003eConex\u00f5es: 19","PRK\u003cbr\u003eConex\u00f5es: 1","LAO\u003cbr\u003eConex\u00f5es: 23","LSO\u003cbr\u003eConex\u00f5es: 3","LBY\u003cbr\u003eConex\u00f5es: 25","MAC\u003cbr\u003eConex\u00f5es: 22","MDG\u003cbr\u003eConex\u00f5es: 42","MWI\u003cbr\u003eConex\u00f5es: 12","MDV\u003cbr\u003eConex\u00f5es: 25","MLI\u003cbr\u003eConex\u00f5es: 21","MRT\u003cbr\u003eConex\u00f5es: 19","MUS\u003cbr\u003eConex\u00f5es: 61","MOZ\u003cbr\u003eConex\u00f5es: 27","NAM\u003cbr\u003eConex\u00f5es: 60","NPL\u003cbr\u003eConex\u00f5es: 15","ABW\u003cbr\u003eConex\u00f5es: 15","VUT\u003cbr\u003eConex\u00f5es: 11","NIC\u003cbr\u003eConex\u00f5es: 34","RWA\u003cbr\u003eConex\u00f5es: 15","KNA\u003cbr\u003eConex\u00f5es: 11","SLE\u003cbr\u003eConex\u00f5es: 14","SOM\u003cbr\u003eConex\u00f5es: 15","ZWE\u003cbr\u003eConex\u00f5es: 16","SDN\u003cbr\u003eConex\u00f5es: 17","SUR\u003cbr\u003eConex\u00f5es: 18","SWZ\u003cbr\u003eConex\u00f5es: 9","TJK\u003cbr\u003eConex\u00f5es: 13","TON\u003cbr\u003eConex\u00f5es: 4","TKM\u003cbr\u003eConex\u00f5es: 12","UGA\u003cbr\u003eConex\u00f5es: 20","UZB\u003cbr\u003eConex\u00f5es: 46","WSM\u003cbr\u003eConex\u00f5es: 5","YEM\u003cbr\u003eConex\u00f5es: 15","KIR\u003cbr\u003eConex\u00f5es: 5","TUV\u003cbr\u003eConex\u00f5es: 3","WLF\u003cbr\u003eConex\u00f5es: 6","CAF\u003cbr\u003eConex\u00f5es: 10","ATF\u003cbr\u003eConex\u00f5es: 3","SXM\u003cbr\u003eConex\u00f5es: 10","GNB\u003cbr\u003eConex\u00f5es: 8","SYC\u003cbr\u003eConex\u00f5es: 23","GUM\u003cbr\u003eConex\u00f5es: 12","FSM\u003cbr\u003eConex\u00f5es: 9","BES\u003cbr\u003eConex\u00f5es: 4","COK\u003cbr\u003eConex\u00f5es: 7","NIU\u003cbr\u003eConex\u00f5es: 4","AFG\u003cbr\u003eConex\u00f5es: 14","BTN\u003cbr\u003eConex\u00f5es: 9","ERI\u003cbr\u003eConex\u00f5es: 4","VAT\u003cbr\u003eConex\u00f5es: 1","LCA\u003cbr\u003eConex\u00f5es: 12","SYR\u003cbr\u003eConex\u00f5es: 7","FLK\u003cbr\u003eConex\u00f5es: 3","SHN\u003cbr\u003eConex\u00f5es: 1","VGB\u003cbr\u003eConex\u00f5es: 12","AIA\u003cbr\u003eConex\u00f5es: 5","UMI\u003cbr\u003eConex\u00f5es: 1","BLM\u003cbr\u003eConex\u00f5es: 4","SSD\u003cbr\u003eConex\u00f5es: 11","IOT\u003cbr\u003eConex\u00f5es: 1","ASM\u003cbr\u003eConex\u00f5es: 6","MSR\u003cbr\u003eConex\u00f5es: 2","MNP\u003cbr\u003eConex\u00f5es: 4","NFK\u003cbr\u003eConex\u00f5es: 1","ATA\u003cbr\u003eConex\u00f5es: 3","PCN\u003cbr\u003eConex\u00f5es: 1"],"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[52.132633,51.919438,null,52.132633,51.165691,null,52.132633,46.603354,null,23.634501,37.09024,null,20.593684,23.424076,null,37.09024,23.634501,null,52.132633,40.463667,null,51.165691,46.603354,null,52.132633,50.850346,null,38.963745,37.09024,null,37.09024,56.130366,null,18.735693,37.09024,null,51.165691,51.919438,null,51.919438,51.165691,null,15.870032,37.09024,null,30.375321,37.09024,null,51.165691,47.516231,null,31.791702,40.463667,null,51.165691,40.463667,null,51.165691,46.818188,null,20.593684,37.09024,null,55.378051,31.791702,null,51.165691,41.87194,null,51.165691,52.132633,null,51.165691,49.817492,null,38.963745,51.165691,null,40.463667,39.399872,null,56.130366,37.09024,null,51.165691,50.850346,null,40.463667,46.603354,null,51.919438,49.817492,null,20.593684,9.081999,null,49.817492,51.165691,null,46.603354,51.165691,null,63.397768,60.472024,null,42.315407,51.919438,null,46.603354,41.87194,null,52.132633,37.09024,null,51.165691,63.397768,null,30.375321,51.165691,null,40.463667,41.87194,null,46.603354,40.463667,null,4.210484,38.963745,null,20.593684,14.497401,null,38.963745,52.132633,null,51.919438,52.132633,null,50.850346,51.165691,null,21.916221,37.09024,null,40.463667,31.791702,null,35.907757,35.86166,null,51.919438,46.603354,null,31.791702,55.378051,null,-23.442503,-14.235004,null,20.593684,23.885942,null,56.26392,51.165691,null,50.850346,46.603354,null,63.397768,51.919438,null,21.916221,35.907757,null,52.132633,41.87194,null,37.09024,51.165691,null,51.919438,48.669026,null,39.399872,40.463667,null,51.165691,55.169438,null,38.963745,47.411631,null,38.963745,40.463667,null,36.204824,14.058324,null,49.817492,51.919438,null,48.379433,48.669026,null,37.09024,23.424076,null,46.603354,33.886917,null,45.943161,51.919438,null,63.397768,56.26392,null,50.850346,52.132633,null,4.210484,46.603354,null,51.165691,33.886917,null,46.603354,52.132633,null,42.315407,49.817492,null,51.165691,31.791702,null,51.165691,56.26392,null,46.603354,48.669026,null,51.919438,48.379433,null,52.132633,53.41291,null,63.397768,61.92411,null,51.165691,47.162494,null,38.963745,41.87194,null,39.399872,46.603354,null,48.379433,51.165691,null,41.87194,46.603354,null,15.870032,49.817492,null,20.593684,-6.369028,null,45.943161,51.165691,null,36.204824,35.86166,null,47.162494,41.87194,null,49.817492,48.669026,null,51.919438,47.162494,null,20.593684,35.86166,null,46.603354,50.850346,null,39.399872,51.165691,null,15.870032,51.165691,null,45.943161,38.963745,null,56.26392,52.132633,null,41.87194,40.463667,null,56.26392,46.603354,null,51.165691,55.378051,null,52.132633,20.593684,null,1.352083,-0.789275,null,22.396428,37.09024,null,55.378051,51.165691,null,49.817492,46.603354,null,48.669026,47.516231,null,20.593684,-0.023559,null,39.399872,49.817492,null,46.603354,45.943161,null,38.963745,39.074208,null,35.907757,36.204824,null,20.593684,7.873054,null,41.87194,48.669026,null,20.593684,8.619543,null,20.593684,55.378051,null,37.09024,55.378051,null,39.399872,48.669026,null,39.399872,55.378051,null,41.377491,38.861034,null,50.850346,40.463667,null,45.943161,46.603354,null,51.919438,45.943161,null,51.919438,56.26392,null,51.919438,63.397768,null,46.603354,46.818188,null,49.817492,37.09024,null,47.162494,51.165691,null,4.210484,1.352083,null,38.963745,28.0,null,35.907757,14.058324,null,15.870032,-25.274398,null,52.132633,56.26392,null,51.919438,40.463667,null,22.396428,35.86166,null,20.593684,46.603354,null,30.375321,55.378051,null,37.09024,35.86166,null,20.593684,15.870032,null,51.919438,50.850346,null,51.919438,61.92411,null,20.593684,21.512583,null,37.09024,-25.274398,null,63.397768,46.603354,null,51.919438,55.378051,null,48.669026,31.791702,null,38.963745,42.733883,null,20.593684,36.204824,null,45.943161,42.733883,null,15.870032,55.378051,null,52.132633,38.963745,null,51.165691,37.09024,null,38.963745,47.162494,null,20.593684,56.130366,null,55.378051,47.162494,null,55.378051,53.41291,null,45.943161,41.87194,null,51.919438,41.87194,null,37.09024,50.850346,null,20.593684,51.165691,null,40.069099,61.52401,null,48.379433,45.943161,null,47.411631,48.669026,null,15.870032,38.963745,null,38.963745,46.603354,null,51.919438,55.169438,null,38.963745,26.3351,null,52.132633,49.817492,null,51.165691,45.943161,null,45.943161,39.074208,null,51.919438,47.516231,null,38.963745,26.820553,null,-25.274398,-40.900557,null,46.603354,51.919438,null,12.879721,36.204824,null,51.165691,53.41291,null,50.850346,49.817492,null,12.879721,15.870032,null,38.963745,45.943161,null,52.132633,-25.274398,null,46.603354,39.399872,null,30.375321,52.132633,null,51.165691,39.399872,null,38.963745,63.397768,null,20.593684,29.31166,null,40.463667,51.165691,null,47.162494,49.817492,null,35.907757,-0.789275,null,38.963745,35.86166,null,31.791702,39.399872,null,20.593684,-25.274398,null,35.907757,22.396428,null,42.315407,50.850346,null,4.210484,-25.274398,null,22.396428,14.058324,null,36.204824,50.850346,null,51.165691,61.92411,null,47.411631,45.943161,null,41.87194,51.165691,null,38.963745,32.427908,null,38.963745,6.428055,null,-30.559482,-2.1646,null,20.593684,12.238333,null,44.016521,51.165691,null,47.162494,48.379433,null,47.162494,44.016521,null,-23.442503,-38.416097,null,30.375321,4.210484,null,38.963745,42.315407,null,42.733883,51.165691,null,15.870032,52.132633,null,51.165691,48.669026,null,52.132633,47.516231,null,51.165691,28.0,null,51.919438,46.818188,null,56.26392,60.472024,null,55.378051,36.204824,null,51.165691,49.815273,null,46.603354,37.09024,null,55.169438,63.397768,null,37.09024,36.204824,null,30.375321,50.850346,null,4.570868,23.634501,null,15.870032,1.352083,null,45.1,51.165691,null,38.963745,51.919438,null,51.165691,61.52401,null,52.132633,63.397768,null,52.132633,55.378051,null,23.634501,8.537981,null,51.919438,56.879635,null,38.963745,31.791702,null,30.375321,46.603354,null,50.850346,51.919438,null,51.165691,23.634501,null,15.870032,36.204824,null,36.204824,61.52401,null,35.907757,37.09024,null,55.378051,-30.559482,null,13.794185,4.570868,null,52.132633,46.818188,null,58.595272,61.92411,null,55.378051,1.352083,null,56.26392,4.860416,null,4.210484,12.879721,null,20.593684,-30.559482,null,40.463667,51.919438,null,51.165691,38.963745,null,49.817492,47.411631,null,20.593684,40.463667,null,45.943161,47.516231,null,56.879635,63.397768,null,52.132633,56.130366,null,46.603354,55.378051,null,55.378051,41.87194,null,41.87194,45.943161,null,49.817492,40.463667,null,51.165691,45.1,null,30.375321,33.0,null,37.09024,52.132633,null,42.315407,31.046051,null,38.963745,55.378051,null,55.378051,46.603354,null,35.907757,23.885942,null,49.817492,63.397768,null,47.162494,48.669026,null,38.963745,39.399872,null,-14.235004,23.634501,null,41.377491,61.52401,null,48.669026,49.817492,null,4.210484,-11.202692,null,51.165691,46.151241,null,51.919438,60.472024,null,63.397768,56.879635,null,52.132633,23.424076,null,56.26392,40.463667,null,41.87194,46.818188,null,35.907757,1.352083,null,30.375321,-25.274398,null,56.26392,63.397768,null,37.09024,6.42375,null,15.870032,40.463667,null,48.019573,61.52401,null,40.463667,-35.675147,null,30.375321,42.733883,null,38.963745,31.046051,null,-25.274398,17.060816,null,4.210484,20.593684,null,4.210484,36.204824,null,38.963745,44.016521,null,4.210484,22.396428,null,55.378051,52.132633,null,37.09024,35.907757,null,23.634501,4.570868,null,38.963745,33.223191,null,49.817492,41.87194,null,31.791702,51.165691,null,20.593684,9.945587,null,51.165691,39.074208,null,48.379433,51.919438,null,41.87194,52.132633,null,39.399872,51.919438,null,38.963745,30.585164,null,38.963745,23.885942,null,41.87194,39.074208,null,56.26392,47.516231,null,49.817492,47.516231,null,20.593684,15.552727,null,51.919438,37.09024,null,22.396428,36.204824,null,42.733883,47.162494,null,4.210484,15.870032,null,52.132633,36.204824,null,40.463667,55.378051,null,12.879721,35.86166,null,63.397768,51.165691,null,42.733883,45.943161,null,13.794185,15.783471,null,47.162494,61.92411,null,20.593684,33.223191,null,37.09024,1.352083,null,42.315407,61.52401,null,37.09024,18.735693,null,56.879635,55.169438,null,56.879635,58.595272,null,47.162494,45.1,null,15.870032,12.565679,null,37.09024,19.3133,null,30.375321,17.607789,null,31.046051,36.204824,null,20.593684,11.825138,null,49.817492,23.885942,null,53.41291,55.378051,null,48.669026,46.603354,null,61.92411,63.397768,null,22.396428,35.907757,null,-0.789275,37.09024,null,36.204824,35.907757,null,51.165691,-30.559482,null,52.132633,61.92411,null,40.463667,4.570868,null,42.315407,41.87194,null,23.885942,26.02751,null,56.26392,50.850346,null,44.016521,49.817492,null,38.963745,49.817492,null,40.463667,23.634501,null,55.378051,37.09024,null,12.879721,37.09024,null,52.132633,48.669026,null,35.907757,10.691803,null,55.169438,51.919438,null,43.915886,51.165691,null,63.397768,55.378051,null,41.87194,55.378051,null,47.162494,45.943161,null,52.132633,64.963051,null,47.162494,33.886917,null,63.397768,40.463667,null,36.204824,37.09024,null,49.817492,50.850346,null,37.09024,-14.235004,null,50.850346,63.397768,null,20.593684,41.87194,null,51.919438,42.733883,null,41.87194,46.151241,null,1.352083,36.204824,null,52.132633,-14.235004,null,20.593684,4.210484,null,37.09024,-1.831239,null,31.046051,37.09024,null,23.634501,51.165691,null,38.963745,-25.274398,null,46.603354,-11.202692,null,12.879721,20.593684,null,55.169438,56.26392,null,15.870032,46.603354,null,55.169438,51.165691,null,45.943161,45.1,null,46.603354,9.945587,null,4.210484,35.907757,null,50.850346,47.516231,null,1.352083,-25.274398,null,20.593684,23.685,null,47.162494,51.919438,null,52.132633,39.399872,null,52.132633,35.86166,null,4.210484,14.058324,null,46.603354,63.397768,null,51.165691,60.472024,null,46.603354,36.204824,null,15.870032,21.00789,null,38.963745,50.850346,null,49.817492,56.26392,null,45.943161,40.463667,null,20.593684,52.132633,null,55.169438,58.595272,null,20.593684,21.916221,null,1.352083,4.210484,null,51.165691,48.379433,null,38.963745,34.802075,null,35.907757,4.860416,null,48.379433,39.399872,null,48.669026,51.919438,null,41.87194,33.886917,null,45.943161,48.669026,null,23.634501,50.850346,null,49.817492,46.818188,null,45.943161,44.016521,null,41.87194,41.0,null,47.162494,46.603354,null,30.375321,56.130366,null,41.87194,26.3351,null,15.870032,23.634501,null,37.09024,-35.675147,null,-0.789275,51.165691,null,46.603354,49.817492,null,56.26392,53.41291,null,23.634501,13.794185,null,38.963745,4.210484,null,38.963745,55.169438,null,46.603354,49.815273,null,31.046051,20.593684,null,30.375321,23.424076,null,51.165691,42.733883,null,47.162494,47.516231,null,20.593684,50.850346,null,23.634501,9.748917,null,44.016521,26.820553,null,41.608635,51.165691,null,30.375321,40.463667,null,49.817492,52.132633,null,46.603354,53.41291,null,56.130366,19.85627,null,55.378051,20.593684,null,41.87194,37.09024,null,52.132633,23.634501,null,38.963745,48.669026,null,35.907757,-32.522779,null,63.397768,39.399872,null,31.046051,40.463667,null,-35.675147,-14.235004,null,37.09024,20.593684,null,1.352083,15.870032,null,49.817492,47.162494,null,55.169438,56.879635,null,44.016521,41.87194,null,46.818188,51.165691,null,36.204824,12.879721,null,-30.559482,-22.95764,null,47.411631,51.165691,null,22.396428,51.165691,null,50.850346,28.0,null,20.593684,28.0,null,45.943161,46.151241,null,55.378051,40.463667,null,30.375321,41.87194,null,39.399872,50.850346,null,42.733883,39.074208,null,-25.274398,37.09024,null,56.879635,51.165691,null,51.919438,58.595272,null,31.046051,45.943161,null,41.87194,49.817492,null,40.463667,52.132633,null,15.870032,7.369722,null,55.378051,-11.202692,null,51.165691,58.595272,null,26.820553,23.424076,null,35.907757,12.879721,null,41.87194,51.919438,null,42.733883,38.963745,null,40.463667,47.162494,null,23.634501,15.783471,null,40.463667,50.850346,null,37.09024,23.885942,null,22.396428,55.378051,null,20.593684,60.472024,null,55.378051,35.86166,null,50.850346,-38.416097,null,56.26392,37.09024,null,51.919438,53.41291,null,36.204824,22.396428,null,45.1,44.016521,null,20.593684,-0.789275,null,46.603354,-20.904305,null,20.593684,35.907757,null,51.165691,44.016521,null,41.87194,50.850346,null,37.09024,9.748917,null,23.885942,25.354826,null,40.463667,45.943161,null,44.016521,48.669026,null,40.463667,49.817492,null,46.603354,47.162494,null,55.378051,50.850346,null,55.378051,51.919438,null,50.850346,55.378051,null,38.963745,35.126413,null,45.943161,50.850346,null,13.794185,37.09024,null,51.919438,46.151241,null,41.87194,47.516231,null,36.204824,15.870032,null,46.603354,47.516231,null,37.09024,4.570868,null,38.963745,42.708678,null,38.963745,40.143105,null,50.850346,15.870032,null,48.669026,47.162494,null,51.165691,56.130366,null,52.132633,45.943161,null,45.943161,52.132633,null,38.963745,43.915886,null,37.09024,53.41291,null,42.315407,48.669026,null,42.315407,40.463667,null,46.603354,14.497401,null,44.016521,38.963745,null,48.379433,47.411631,null,41.87194,39.399872,null,51.919438,61.52401,null,38.963745,53.709807,null,-40.900557,-25.274398,null,15.870032,56.130366,null,44.016521,45.943161,null,63.397768,36.204824,null,37.09024,46.818188,null,39.399872,18.735693,null,4.210484,61.52401,null,50.850346,41.87194,null,42.315407,46.603354,null,39.399872,53.41291,null,15.870032,14.058324,null,63.397768,46.818188,null,61.92411,58.595272,null,-17.713371,-7.109535,null,44.016521,50.850346,null,37.09024,25.354826,null,45.1,43.915886,null,46.151241,45.1,null,37.09024,22.396428,null,42.315407,52.132633,null,56.26392,41.87194,null,35.907757,-20.904305,null,23.885942,8.460555,null,63.397768,64.963051,null,49.817492,55.378051,null,46.603354,31.791702,null,13.794185,-9.189967,null,50.850346,49.815273,null,37.09024,-6.369028,null,38.963745,33.886917,null,55.169438,60.472024,null,40.463667,37.09024,null,38.963745,30.375321,null,51.165691,56.879635,null,35.907757,21.916221,null,51.165691,35.86166,null,42.315407,47.162494,null],"line":{"color":"rgba(0, 150, 0, 0.7)","width":0.5009529334590587},"lon":[5.291266,19.145136,null,5.291266,10.451526,null,5.291266,1.888334,null,-102.552784,-95.712891,null,78.96288,53.847818,null,-95.712891,-102.552784,null,5.291266,-3.74922,null,10.451526,1.888334,null,5.291266,4.351721,null,35.243322,-95.712891,null,-95.712891,-106.346771,null,-70.162651,-95.712891,null,10.451526,19.145136,null,19.145136,10.451526,null,100.992541,-95.712891,null,69.345116,-95.712891,null,10.451526,14.550072,null,-7.09262,-3.74922,null,10.451526,-3.74922,null,10.451526,8.227512,null,78.96288,-95.712891,null,-3.435973,-7.09262,null,10.451526,12.56738,null,10.451526,5.291266,null,10.451526,15.472962,null,35.243322,10.451526,null,-3.74922,-8.224454,null,-106.346771,-95.712891,null,10.451526,4.351721,null,-3.74922,1.888334,null,19.145136,15.472962,null,78.96288,8.675277,null,15.472962,10.451526,null,1.888334,10.451526,null,16.354896,8.468946,null,43.356892,19.145136,null,1.888334,12.56738,null,5.291266,-95.712891,null,10.451526,16.354896,null,69.345116,10.451526,null,-3.74922,12.56738,null,1.888334,-3.74922,null,101.975766,35.243322,null,78.96288,-14.452362,null,35.243322,5.291266,null,19.145136,5.291266,null,4.351721,10.451526,null,95.955974,-95.712891,null,-3.74922,-7.09262,null,127.766922,104.195397,null,19.145136,1.888334,null,-7.09262,-3.435973,null,-58.443832,-51.92528,null,78.96288,45.079162,null,9.501785,10.451526,null,4.351721,1.888334,null,16.354896,19.145136,null,95.955974,127.766922,null,5.291266,12.56738,null,-95.712891,10.451526,null,19.145136,19.699024,null,-8.224454,-3.74922,null,10.451526,23.881275,null,35.243322,28.369885,null,35.243322,-3.74922,null,138.252924,108.277199,null,15.472962,19.145136,null,31.16558,19.699024,null,-95.712891,53.847818,null,1.888334,9.537499,null,24.96676,19.145136,null,16.354896,9.501785,null,4.351721,5.291266,null,101.975766,1.888334,null,10.451526,9.537499,null,1.888334,5.291266,null,43.356892,15.472962,null,10.451526,-7.09262,null,10.451526,9.501785,null,1.888334,19.699024,null,19.145136,31.16558,null,5.291266,-8.24389,null,16.354896,25.748151,null,10.451526,19.503304,null,35.243322,12.56738,null,-8.224454,1.888334,null,31.16558,10.451526,null,12.56738,1.888334,null,100.992541,15.472962,null,78.96288,34.888822,null,24.96676,10.451526,null,138.252924,104.195397,null,19.503304,12.56738,null,15.472962,19.699024,null,19.145136,19.503304,null,78.96288,104.195397,null,1.888334,4.351721,null,-8.224454,10.451526,null,100.992541,10.451526,null,24.96676,35.243322,null,9.501785,5.291266,null,12.56738,-3.74922,null,9.501785,1.888334,null,10.451526,-3.435973,null,5.291266,78.96288,null,103.819836,113.921327,null,114.109497,-95.712891,null,-3.435973,10.451526,null,15.472962,1.888334,null,19.699024,14.550072,null,78.96288,37.906193,null,-8.224454,15.472962,null,1.888334,24.96676,null,35.243322,21.824312,null,127.766922,138.252924,null,78.96288,80.771797,null,12.56738,19.699024,null,78.96288,0.824782,null,78.96288,-3.435973,null,-95.712891,-3.435973,null,-8.224454,19.699024,null,-8.224454,-3.435973,null,64.585262,71.276093,null,4.351721,-3.74922,null,24.96676,1.888334,null,19.145136,24.96676,null,19.145136,9.501785,null,19.145136,16.354896,null,1.888334,8.227512,null,15.472962,-95.712891,null,19.503304,10.451526,null,101.975766,103.819836,null,35.243322,3.0,null,127.766922,108.277199,null,100.992541,133.775136,null,5.291266,9.501785,null,19.145136,-3.74922,null,114.109497,104.195397,null,78.96288,1.888334,null,69.345116,-3.435973,null,-95.712891,104.195397,null,78.96288,100.992541,null,19.145136,4.351721,null,19.145136,25.748151,null,78.96288,55.923255,null,-95.712891,133.775136,null,16.354896,1.888334,null,19.145136,-3.435973,null,19.699024,-7.09262,null,35.243322,25.48583,null,78.96288,138.252924,null,24.96676,25.48583,null,100.992541,-3.435973,null,5.291266,35.243322,null,10.451526,-95.712891,null,35.243322,19.503304,null,78.96288,-106.346771,null,-3.435973,19.503304,null,-3.435973,-8.24389,null,24.96676,12.56738,null,19.145136,12.56738,null,-95.712891,4.351721,null,78.96288,10.451526,null,45.038189,105.318756,null,31.16558,24.96676,null,28.369885,19.699024,null,100.992541,35.243322,null,35.243322,1.888334,null,19.145136,23.881275,null,35.243322,17.228331,null,5.291266,15.472962,null,10.451526,24.96676,null,24.96676,21.824312,null,19.145136,14.550072,null,35.243322,30.802498,null,133.775136,174.885971,null,1.888334,19.145136,null,121.774017,138.252924,null,10.451526,-8.24389,null,4.351721,15.472962,null,121.774017,100.992541,null,35.243322,24.96676,null,5.291266,133.775136,null,1.888334,-8.224454,null,69.345116,5.291266,null,10.451526,-8.224454,null,35.243322,16.354896,null,78.96288,47.481766,null,-3.74922,10.451526,null,19.503304,15.472962,null,127.766922,113.921327,null,35.243322,104.195397,null,-7.09262,-8.224454,null,78.96288,133.775136,null,127.766922,114.109497,null,43.356892,4.351721,null,101.975766,133.775136,null,114.109497,108.277199,null,138.252924,4.351721,null,10.451526,25.748151,null,28.369885,24.96676,null,12.56738,10.451526,null,35.243322,53.688046,null,35.243322,-9.429499,null,22.937506,24.15536,null,78.96288,-1.561593,null,21.005859,10.451526,null,19.503304,31.16558,null,19.503304,21.005859,null,-58.443832,-63.616672,null,69.345116,101.975766,null,35.243322,43.356892,null,25.48583,10.451526,null,100.992541,5.291266,null,10.451526,19.699024,null,5.291266,14.550072,null,10.451526,3.0,null,19.145136,8.227512,null,9.501785,8.468946,null,-3.435973,138.252924,null,10.451526,6.129583,null,1.888334,-95.712891,null,23.881275,16.354896,null,-95.712891,138.252924,null,69.345116,4.351721,null,-74.297333,-102.552784,null,100.992541,103.819836,null,15.2,10.451526,null,35.243322,19.145136,null,10.451526,105.318756,null,5.291266,16.354896,null,5.291266,-3.435973,null,-102.552784,-80.782127,null,19.145136,24.603189,null,35.243322,-7.09262,null,69.345116,1.888334,null,4.351721,19.145136,null,10.451526,-102.552784,null,100.992541,138.252924,null,138.252924,105.318756,null,127.766922,-95.712891,null,-3.435973,22.937506,null,-88.89653,-74.297333,null,5.291266,8.227512,null,25.013607,25.748151,null,-3.435973,103.819836,null,9.501785,-58.93018,null,101.975766,121.774017,null,78.96288,22.937506,null,-3.74922,19.145136,null,10.451526,35.243322,null,15.472962,28.369885,null,78.96288,-3.74922,null,24.96676,14.550072,null,24.603189,16.354896,null,5.291266,-106.346771,null,1.888334,-3.435973,null,-3.435973,12.56738,null,12.56738,24.96676,null,15.472962,-3.74922,null,10.451526,15.2,null,69.345116,65.0,null,-95.712891,5.291266,null,43.356892,34.851612,null,35.243322,-3.435973,null,-3.435973,1.888334,null,127.766922,45.079162,null,15.472962,16.354896,null,19.503304,19.699024,null,35.243322,-8.224454,null,-51.92528,-102.552784,null,64.585262,105.318756,null,19.699024,15.472962,null,101.975766,17.873887,null,10.451526,14.995463,null,19.145136,8.468946,null,16.354896,24.603189,null,5.291266,53.847818,null,9.501785,-3.74922,null,12.56738,8.227512,null,127.766922,103.819836,null,69.345116,133.775136,null,9.501785,16.354896,null,-95.712891,-66.58973,null,100.992541,-3.74922,null,66.923684,105.318756,null,-3.74922,-71.542969,null,69.345116,25.48583,null,35.243322,34.851612,null,133.775136,-61.796428,null,101.975766,78.96288,null,101.975766,138.252924,null,35.243322,21.005859,null,101.975766,114.109497,null,-3.435973,5.291266,null,-95.712891,127.766922,null,-102.552784,-74.297333,null,35.243322,43.679291,null,15.472962,12.56738,null,-7.09262,10.451526,null,78.96288,-9.696645,null,10.451526,21.824312,null,31.16558,19.145136,null,12.56738,5.291266,null,-8.224454,19.145136,null,35.243322,36.238414,null,35.243322,45.079162,null,12.56738,21.824312,null,9.501785,14.550072,null,15.472962,14.550072,null,78.96288,48.516388,null,19.145136,-95.712891,null,114.109497,138.252924,null,25.48583,19.503304,null,101.975766,100.992541,null,5.291266,138.252924,null,-3.74922,-3.435973,null,121.774017,104.195397,null,16.354896,10.451526,null,25.48583,24.96676,null,-88.89653,-90.230759,null,19.503304,25.748151,null,78.96288,43.679291,null,-95.712891,103.819836,null,43.356892,105.318756,null,-95.712891,-70.162651,null,24.603189,23.881275,null,24.603189,25.013607,null,19.503304,15.2,null,100.992541,104.990963,null,-95.712891,-81.2546,null,69.345116,8.081666,null,34.851612,138.252924,null,78.96288,42.590275,null,15.472962,45.079162,null,-8.24389,-3.435973,null,19.699024,1.888334,null,25.748151,16.354896,null,114.109497,127.766922,null,113.921327,-95.712891,null,138.252924,127.766922,null,10.451526,22.937506,null,5.291266,25.748151,null,-3.74922,-74.297333,null,43.356892,12.56738,null,45.079162,50.55096,null,9.501785,4.351721,null,21.005859,15.472962,null,35.243322,15.472962,null,-3.74922,-102.552784,null,-3.435973,-95.712891,null,121.774017,-95.712891,null,5.291266,19.699024,null,127.766922,-61.222503,null,23.881275,19.145136,null,17.679076,10.451526,null,16.354896,-3.435973,null,12.56738,-3.435973,null,19.503304,24.96676,null,5.291266,-19.020835,null,19.503304,9.537499,null,16.354896,-3.74922,null,138.252924,-95.712891,null,15.472962,4.351721,null,-95.712891,-51.92528,null,4.351721,16.354896,null,78.96288,12.56738,null,19.145136,25.48583,null,12.56738,14.995463,null,103.819836,138.252924,null,5.291266,-51.92528,null,78.96288,101.975766,null,-95.712891,-78.183406,null,34.851612,-95.712891,null,-102.552784,10.451526,null,35.243322,133.775136,null,1.888334,17.873887,null,121.774017,78.96288,null,23.881275,9.501785,null,100.992541,1.888334,null,23.881275,10.451526,null,24.96676,15.2,null,1.888334,-9.696645,null,101.975766,127.766922,null,4.351721,14.550072,null,103.819836,133.775136,null,78.96288,90.3563,null,19.503304,19.145136,null,5.291266,-8.224454,null,5.291266,104.195397,null,101.975766,108.277199,null,1.888334,16.354896,null,10.451526,8.468946,null,1.888334,138.252924,null,100.992541,-10.940835,null,35.243322,4.351721,null,15.472962,9.501785,null,24.96676,-3.74922,null,78.96288,5.291266,null,23.881275,25.013607,null,78.96288,95.955974,null,103.819836,101.975766,null,10.451526,31.16558,null,35.243322,38.996815,null,127.766922,-58.93018,null,31.16558,-8.224454,null,19.699024,19.145136,null,12.56738,9.537499,null,24.96676,19.699024,null,-102.552784,4.351721,null,15.472962,8.227512,null,24.96676,21.005859,null,12.56738,20.0,null,19.503304,1.888334,null,69.345116,-106.346771,null,12.56738,17.228331,null,100.992541,-102.552784,null,-95.712891,-71.542969,null,113.921327,10.451526,null,1.888334,15.472962,null,9.501785,-8.24389,null,-102.552784,-88.89653,null,35.243322,101.975766,null,35.243322,23.881275,null,1.888334,6.129583,null,34.851612,78.96288,null,69.345116,53.847818,null,10.451526,25.48583,null,19.503304,14.550072,null,78.96288,4.351721,null,-102.552784,-83.753428,null,21.005859,30.802498,null,21.745275,10.451526,null,69.345116,-3.74922,null,15.472962,5.291266,null,1.888334,-8.24389,null,-106.346771,102.495496,null,-3.435973,78.96288,null,12.56738,-95.712891,null,5.291266,-102.552784,null,35.243322,19.699024,null,127.766922,-55.765835,null,16.354896,-8.224454,null,34.851612,-3.74922,null,-71.542969,-51.92528,null,-95.712891,78.96288,null,103.819836,100.992541,null,15.472962,19.503304,null,23.881275,24.603189,null,21.005859,12.56738,null,8.227512,10.451526,null,138.252924,121.774017,null,22.937506,18.49041,null,28.369885,10.451526,null,114.109497,10.451526,null,4.351721,3.0,null,78.96288,3.0,null,24.96676,14.995463,null,-3.435973,-3.74922,null,69.345116,12.56738,null,-8.224454,4.351721,null,25.48583,21.824312,null,133.775136,-95.712891,null,24.603189,10.451526,null,19.145136,25.013607,null,34.851612,24.96676,null,12.56738,15.472962,null,-3.74922,5.291266,null,100.992541,12.354722,null,-3.435973,17.873887,null,10.451526,25.013607,null,30.802498,53.847818,null,127.766922,121.774017,null,12.56738,19.145136,null,25.48583,35.243322,null,-3.74922,19.503304,null,-102.552784,-90.230759,null,-3.74922,4.351721,null,-95.712891,45.079162,null,114.109497,-3.435973,null,78.96288,8.468946,null,-3.435973,104.195397,null,4.351721,-63.616672,null,9.501785,-95.712891,null,19.145136,-8.24389,null,138.252924,114.109497,null,15.2,21.005859,null,78.96288,113.921327,null,1.888334,165.618042,null,78.96288,127.766922,null,10.451526,21.005859,null,12.56738,4.351721,null,-95.712891,-83.753428,null,45.079162,51.183884,null,-3.74922,24.96676,null,21.005859,19.699024,null,-3.74922,15.472962,null,1.888334,19.503304,null,-3.435973,4.351721,null,-3.435973,19.145136,null,4.351721,-3.435973,null,35.243322,33.429859,null,24.96676,4.351721,null,-88.89653,-95.712891,null,19.145136,14.995463,null,12.56738,14.550072,null,138.252924,100.992541,null,1.888334,14.550072,null,-95.712891,-74.297333,null,35.243322,19.37439,null,35.243322,47.576927,null,4.351721,100.992541,null,19.699024,19.503304,null,10.451526,-106.346771,null,5.291266,24.96676,null,24.96676,5.291266,null,35.243322,17.679076,null,-95.712891,-8.24389,null,43.356892,19.699024,null,43.356892,-3.74922,null,1.888334,-14.452362,null,21.005859,35.243322,null,31.16558,28.369885,null,12.56738,-8.224454,null,19.145136,105.318756,null,35.243322,27.953389,null,174.885971,133.775136,null,100.992541,-106.346771,null,21.005859,24.96676,null,16.354896,138.252924,null,-95.712891,8.227512,null,-8.224454,-70.162651,null,101.975766,105.318756,null,4.351721,12.56738,null,43.356892,1.888334,null,-8.224454,-8.24389,null,100.992541,108.277199,null,16.354896,8.227512,null,25.748151,25.013607,null,178.065033,179.194167,null,21.005859,4.351721,null,-95.712891,51.183884,null,15.2,17.679076,null,14.995463,15.2,null,-95.712891,114.109497,null,43.356892,5.291266,null,9.501785,12.56738,null,127.766922,165.618042,null,45.079162,-11.779889,null,16.354896,-19.020835,null,15.472962,-3.435973,null,1.888334,-7.09262,null,-88.89653,-75.015152,null,4.351721,6.129583,null,-95.712891,34.888822,null,35.243322,9.537499,null,23.881275,8.468946,null,-3.74922,-95.712891,null,35.243322,69.345116,null,10.451526,24.603189,null,127.766922,95.955974,null,10.451526,104.195397,null,43.356892,19.503304,null],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.5)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hoverinfo":"none","lat":[],"line":{"color":"rgba(0, 150, 0, 0.3)","width":1},"lon":[],"mode":"lines","showlegend":false,"visible":false,"type":"scattergeo"},{"hovertemplate":"%{text}\u003cextra\u003e\u003c\u002fextra\u003e","lat":[42.546245,8.537981,40.463667,-11.202692,50.850346,41.87194,-22.95764,39.399872,23.424076,17.060816,37.09024,-38.416097,-35.675147,-23.442503,-32.522779,-25.274398,-22.328474,4.535277,12.565679,7.873054,35.86166,56.26392,-17.713371,-17.679742,7.946527,39.074208,22.396428,-0.789275,53.41291,7.539989,36.204824,35.907757,4.210484,35.937496,-0.522778,52.132633,-15.376706,-40.900557,-6.314993,12.879721,51.919438,23.885942,20.593684,1.352083,-30.559482,15.870032,55.378051,-6.369028,40.069099,53.709807,12.16957,61.52401,13.193887,13.909444,12.984305,41.0,28.0,40.143105,47.516231,26.02751,-14.235004,42.733883,7.369722,56.130366,15.454166,-0.228021,-2.1646,45.1,35.126413,49.817492,18.735693,-1.831239,1.650801,9.145,58.595272,61.92411,46.603354,-0.803689,42.315407,51.165691,36.140751,47.162494,64.963051,31.046051,48.019573,29.31166,41.20438,33.854721,56.879635,55.169438,49.815273,22.198745,21.00789,-20.348404,23.634501,46.862496,47.411631,31.791702,-20.904305,9.081999,60.472024,-9.189967,25.354826,45.943161,-1.940278,17.897476,43.94236,14.497401,44.016521,-4.679574,48.669026,46.151241,63.397768,46.818188,33.886917,38.963745,1.373333,48.379433,41.608635,26.820553,12.238333,-13.133897,-16.290154,43.915886,25.03428,32.3078,19.3133,4.570868,9.748917,15.783471,18.971187,30.585164,6.428055,7.131474,3.919305,41.377491,6.42375,42.708678,14.058324,21.916221,21.521757,4.860416,18.109581,18.04248,10.691803,16.5388,13.794185,15.199999,38.969719,61.892635,71.706936,-0.023559,21.694025,12.865416,-3.370417,-21.178986,-13.768752,-13.759029,23.685,-11.6455,9.30769,13.444304,9.945587,-18.766947,17.570692,21.512583,12.52111,46.946947,8.460555,8.619543,33.0,32.427908,33.223191,26.3351,3.202778,30.375321,38.861034,27.514162,-26.522503,17.189877,-21.236736,-13.254308,-8.874217,19.85627,-90.0,18.420695,12.20189,-9.64571,-19.054445,-24.376515,-7.109535,-49.280366,-19.015438,-18.665695,11.803749,0.18636,5.152149,28.394857,15.552727,-29.609988,13.443182,12.262776,-3.373056,11.825138,7.862684,34.802075,-51.796253,17.357822,15.414999,6.611111,31.952162,10.961632,17.607789,7.425554,-10.447525,-14.28522,15.179384,12.862807,24.215527,18.218785,17.664332,-29.040835,7.51498,-6.343194,-15.965,16.742498,-9.2],"lon":[1.601554,-80.782127,-3.74922,17.873887,4.351721,12.56738,18.49041,-8.224454,53.847818,-61.796428,-95.712891,-63.616672,-71.542969,-58.443832,-55.765835,133.775136,24.684866,114.727669,104.990963,80.771797,104.195397,9.501785,178.065033,-149.406843,-1.023194,21.824312,114.109497,113.921327,-8.24389,-5.54708,138.252924,127.766922,101.975766,14.375416,166.931503,5.291266,166.959158,174.885971,143.95555,121.774017,19.145136,45.079162,78.96288,103.819836,22.937506,100.992541,-3.435973,34.888822,45.038189,27.953389,-68.990021,105.318756,-59.543198,-60.978893,-61.287228,20.0,3.0,47.576927,14.550072,50.55096,-51.92528,25.48583,12.354722,-106.346771,18.732207,15.827659,24.15536,15.2,33.429859,15.472962,-70.162651,-78.183406,10.267895,40.489673,25.013607,25.748151,1.888334,11.609444,43.356892,10.451526,-5.353585,19.503304,-19.020835,34.851612,66.923684,47.481766,74.766098,35.862285,24.603189,23.881275,6.129583,113.543873,-10.940835,57.552152,-102.552784,103.846656,28.369885,-7.09262,165.618042,8.675277,8.468946,-75.015152,51.183884,24.96676,29.873888,-62.83055,12.457777,-14.452362,21.005859,55.491977,19.699024,14.995463,16.354896,8.227512,9.537499,35.243322,32.290275,31.16558,21.745275,30.802498,-1.561593,27.849332,-63.588653,17.679076,-77.39628,-64.7505,-81.2546,-74.297333,-83.753428,-90.230759,-72.285215,36.238414,-9.429499,171.184478,-56.027783,64.585262,-66.58973,19.37439,108.277199,95.955974,-77.781167,-58.93018,-77.297508,-63.05483,-61.222503,-23.0418,-88.89653,-86.241905,59.556278,-6.911806,-42.604303,37.906193,-71.797928,-85.207229,-168.734039,-175.198242,-177.156097,-172.104629,90.3563,43.3333,2.315834,144.793731,-9.696645,46.869107,-3.996166,55.923255,-69.968338,-56.32509,-11.779889,0.824782,65.0,53.688046,43.679291,17.228331,73.22068,69.345116,71.276093,90.433601,31.465866,-88.49765,-159.777671,34.301525,125.727,102.495496,0.0,-64.639968,-68.262383,160.156194,-169.867233,-128.324001,179.194167,69.348557,29.154857,35.529562,-15.180413,6.613081,46.199616,84.124008,48.516388,28.233608,-15.310139,-61.604171,29.918886,42.590275,30.217636,38.996815,-59.523613,-62.782998,-61.370976,20.939444,35.233154,-169.09022,8.081666,150.550812,105.690449,-170.70444,39.782334,30.217636,-12.885834,-63.043653,145.94351,167.954712,134.58252,71.876519,-5.7089,-62.187366,-171.833333],"marker":{"color":"red","opacity":0.8,"size":[6.4,20.0,20.0,18.4,20.0,20.0,20.0,20.0,20.0,10.4,20.0,16.8,20.0,15.2,13.6,20.0,4.4,8.0,7.6,10.4,20.0,20.0,12.4,7.6,11.6,20.0,20.0,20.0,20.0,11.2,20.0,20.0,20.0,20.0,0.8,20.0,3.6,20.0,6.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,9.2,19.6,8.0,7.6,16.8,14.4,5.2,2.8,12.8,9.6,19.6,19.2,13.6,20.0,20.0,7.2,20.0,3.2,7.6,7.6,20.0,20.0,20.0,20.0,16.0,5.2,8.8,20.0,20.0,20.0,9.2,20.0,20.0,7.6,20.0,16.8,20.0,18.8,16.4,18.0,14.4,20.0,20.0,20.0,8.4,7.6,20.0,20.0,12.4,20.0,20.0,6.0,10.4,20.0,15.2,17.6,20.0,6.0,2.4,2.8,12.4,20.0,8.0,20.0,20.0,20.0,20.0,10.0,20.0,7.2,20.0,20.0,20.0,14.4,6.4,10.4,20.0,9.2,5.6,6.4,20.0,16.0,20.0,4.4,12.4,7.2,5.2,7.6,19.6,8.4,16.8,14.8,13.6,8.4,14.4,9.2,5.6,20.0,6.0,16.0,10.0,6.8,5.6,6.8,11.6,2.4,6.8,2.8,1.6,0.8,2.0,9.6,1.6,5.6,4.4,9.2,6.8,8.8,13.2,5.6,0.4,5.6,6.8,4.4,6.0,9.2,9.6,8.8,20.0,5.2,2.4,3.2,6.8,2.8,3.2,2.8,4.8,1.6,4.4,0.8,3.2,0.4,0.4,1.6,0.8,6.0,6.0,2.4,1.6,5.2,5.2,3.6,1.2,3.6,4.8,3.2,2.8,4.4,1.2,0.8,2.8,4.0,4.0,2.8,1.2,5.2,2.0,0.8,1.2,2.4,4.4,0.8,2.0,1.6,0.8,1.2,0.4,0.8,0.4,0.8]},"mode":"markers","name":"2024","text":["AND\u003cbr\u003eConex\u00f5es: 16","PAN\u003cbr\u003eConex\u00f5es: 56","ESP\u003cbr\u003eConex\u00f5es: 206","AGO\u003cbr\u003eConex\u00f5es: 46","BEL\u003cbr\u003eConex\u00f5es: 223","ITA\u003cbr\u003eConex\u00f5es: 200","NAM\u003cbr\u003eConex\u00f5es: 58","PRT\u003cbr\u003eConex\u00f5es: 183","ARE\u003cbr\u003eConex\u00f5es: 61","ATG\u003cbr\u003eConex\u00f5es: 26","USA\u003cbr\u003eConex\u00f5es: 239","ARG\u003cbr\u003eConex\u00f5es: 42","CHL\u003cbr\u003eConex\u00f5es: 68","PRY\u003cbr\u003eConex\u00f5es: 38","URY\u003cbr\u003eConex\u00f5es: 34","AUS\u003cbr\u003eConex\u00f5es: 154","BWA\u003cbr\u003eConex\u00f5es: 11","BRN\u003cbr\u003eConex\u00f5es: 20","KHM\u003cbr\u003eConex\u00f5es: 19","LKA\u003cbr\u003eConex\u00f5es: 26","CHN\u003cbr\u003eConex\u00f5es: 54","DNK\u003cbr\u003eConex\u00f5es: 163","FJI\u003cbr\u003eConex\u00f5es: 31","PYF\u003cbr\u003eConex\u00f5es: 19","GHA\u003cbr\u003eConex\u00f5es: 29","GRC\u003cbr\u003eConex\u00f5es: 124","HKG\u003cbr\u003eConex\u00f5es: 166","IDN\u003cbr\u003eConex\u00f5es: 116","IRL\u003cbr\u003eConex\u00f5es: 119","CIV\u003cbr\u003eConex\u00f5es: 28","JPN\u003cbr\u003eConex\u00f5es: 146","KOR\u003cbr\u003eConex\u00f5es: 184","MYS\u003cbr\u003eConex\u00f5es: 149","MLT\u003cbr\u003eConex\u00f5es: 61","NRU\u003cbr\u003eConex\u00f5es: 2","NLD\u003cbr\u003eConex\u00f5es: 235","VUT\u003cbr\u003eConex\u00f5es: 9","NZL\u003cbr\u003eConex\u00f5es: 113","PNG\u003cbr\u003eConex\u00f5es: 15","PHL\u003cbr\u003eConex\u00f5es: 72","POL\u003cbr\u003eConex\u00f5es: 193","SAU\u003cbr\u003eConex\u00f5es: 63","IND\u003cbr\u003eConex\u00f5es: 212","SGP\u003cbr\u003eConex\u00f5es: 113","ZAF\u003cbr\u003eConex\u00f5es: 159","THA\u003cbr\u003eConex\u00f5es: 148","GBR\u003cbr\u003eConex\u00f5es: 215","TZA\u003cbr\u003eConex\u00f5es: 23","ARM\u003cbr\u003eConex\u00f5es: 49","BLR\u003cbr\u003eConex\u00f5es: 20","CUW\u003cbr\u003eConex\u00f5es: 19","RUS\u003cbr\u003eConex\u00f5es: 42","BRB\u003cbr\u003eConex\u00f5es: 36","LCA\u003cbr\u003eConex\u00f5es: 13","VCT\u003cbr\u003eConex\u00f5es: 7","ALB\u003cbr\u003eConex\u00f5es: 32","DZA\u003cbr\u003eConex\u00f5es: 24","AZE\u003cbr\u003eConex\u00f5es: 49","AUT\u003cbr\u003eConex\u00f5es: 48","BHR\u003cbr\u003eConex\u00f5es: 34","BRA\u003cbr\u003eConex\u00f5es: 151","BGR\u003cbr\u003eConex\u00f5es: 109","CMR\u003cbr\u003eConex\u00f5es: 18","CAN\u003cbr\u003eConex\u00f5es: 172","TCD\u003cbr\u003eConex\u00f5es: 8","COG\u003cbr\u003eConex\u00f5es: 19","COD\u003cbr\u003eConex\u00f5es: 19","HRV\u003cbr\u003eConex\u00f5es: 95","CYP\u003cbr\u003eConex\u00f5es: 51","CZE\u003cbr\u003eConex\u00f5es: 169","DOM\u003cbr\u003eConex\u00f5es: 53","ECU\u003cbr\u003eConex\u00f5es: 40","GNQ\u003cbr\u003eConex\u00f5es: 13","ETH\u003cbr\u003eConex\u00f5es: 22","EST\u003cbr\u003eConex\u00f5es: 100","FIN\u003cbr\u003eConex\u00f5es: 133","FRA\u003cbr\u003eConex\u00f5es: 198","GAB\u003cbr\u003eConex\u00f5es: 23","GEO\u003cbr\u003eConex\u00f5es: 67","DEU\u003cbr\u003eConex\u00f5es: 248","GIB\u003cbr\u003eConex\u00f5es: 19","HUN\u003cbr\u003eConex\u00f5es: 130","ISL\u003cbr\u003eConex\u00f5es: 42","ISR\u003cbr\u003eConex\u00f5es: 89","KAZ\u003cbr\u003eConex\u00f5es: 47","KWT\u003cbr\u003eConex\u00f5es: 41","KGZ\u003cbr\u003eConex\u00f5es: 45","LBN\u003cbr\u003eConex\u00f5es: 36","LVA\u003cbr\u003eConex\u00f5es: 99","LTU\u003cbr\u003eConex\u00f5es: 122","LUX\u003cbr\u003eConex\u00f5es: 97","MAC\u003cbr\u003eConex\u00f5es: 21","MRT\u003cbr\u003eConex\u00f5es: 19","MUS\u003cbr\u003eConex\u00f5es: 63","MEX\u003cbr\u003eConex\u00f5es: 76","MNG\u003cbr\u003eConex\u00f5es: 31","MDA\u003cbr\u003eConex\u00f5es: 50","MAR\u003cbr\u003eConex\u00f5es: 75","NCL\u003cbr\u003eConex\u00f5es: 15","NGA\u003cbr\u003eConex\u00f5es: 26","NOR\u003cbr\u003eConex\u00f5es: 136","PER\u003cbr\u003eConex\u00f5es: 38","QAT\u003cbr\u003eConex\u00f5es: 44","ROU\u003cbr\u003eConex\u00f5es: 122","RWA\u003cbr\u003eConex\u00f5es: 15","BLM\u003cbr\u003eConex\u00f5es: 6","SMR\u003cbr\u003eConex\u00f5es: 7","SEN\u003cbr\u003eConex\u00f5es: 31","SRB\u003cbr\u003eConex\u00f5es: 92","SYC\u003cbr\u003eConex\u00f5es: 20","SVK\u003cbr\u003eConex\u00f5es: 127","SVN\u003cbr\u003eConex\u00f5es: 117","SWE\u003cbr\u003eConex\u00f5es: 180","CHE\u003cbr\u003eConex\u00f5es: 196","TUN\u003cbr\u003eConex\u00f5es: 25","TUR\u003cbr\u003eConex\u00f5es: 232","UGA\u003cbr\u003eConex\u00f5es: 18","UKR\u003cbr\u003eConex\u00f5es: 111","MKD\u003cbr\u003eConex\u00f5es: 50","EGY\u003cbr\u003eConex\u00f5es: 59","BFA\u003cbr\u003eConex\u00f5es: 36","ZMB\u003cbr\u003eConex\u00f5es: 16","BOL\u003cbr\u003eConex\u00f5es: 26","BIH\u003cbr\u003eConex\u00f5es: 61","BHS\u003cbr\u003eConex\u00f5es: 23","BMU\u003cbr\u003eConex\u00f5es: 14","CYM\u003cbr\u003eConex\u00f5es: 16","COL\u003cbr\u003eConex\u00f5es: 65","CRI\u003cbr\u003eConex\u00f5es: 40","GTM\u003cbr\u003eConex\u00f5es: 53","HTI\u003cbr\u003eConex\u00f5es: 11","JOR\u003cbr\u003eConex\u00f5es: 31","LBR\u003cbr\u003eConex\u00f5es: 18","MHL\u003cbr\u003eConex\u00f5es: 13","SUR\u003cbr\u003eConex\u00f5es: 19","UZB\u003cbr\u003eConex\u00f5es: 49","VEN\u003cbr\u003eConex\u00f5es: 21","MNE\u003cbr\u003eConex\u00f5es: 42","VNM\u003cbr\u003eConex\u00f5es: 37","MMR\u003cbr\u003eConex\u00f5es: 34","CUB\u003cbr\u003eConex\u00f5es: 21","GUY\u003cbr\u003eConex\u00f5es: 36","JAM\u003cbr\u003eConex\u00f5es: 23","SXM\u003cbr\u003eConex\u00f5es: 14","TTO\u003cbr\u003eConex\u00f5es: 52","CPV\u003cbr\u003eConex\u00f5es: 15","SLV\u003cbr\u003eConex\u00f5es: 40","HND\u003cbr\u003eConex\u00f5es: 25","TKM\u003cbr\u003eConex\u00f5es: 17","FRO\u003cbr\u003eConex\u00f5es: 14","GRL\u003cbr\u003eConex\u00f5es: 17","KEN\u003cbr\u003eConex\u00f5es: 29","TCA\u003cbr\u003eConex\u00f5es: 6","NIC\u003cbr\u003eConex\u00f5es: 17","KIR\u003cbr\u003eConex\u00f5es: 7","TON\u003cbr\u003eConex\u00f5es: 4","WLF\u003cbr\u003eConex\u00f5es: 2","WSM\u003cbr\u003eConex\u00f5es: 5","BGD\u003cbr\u003eConex\u00f5es: 24","COM\u003cbr\u003eConex\u00f5es: 4","BEN\u003cbr\u003eConex\u00f5es: 14","GUM\u003cbr\u003eConex\u00f5es: 11","GIN\u003cbr\u003eConex\u00f5es: 23","MDG\u003cbr\u003eConex\u00f5es: 17","MLI\u003cbr\u003eConex\u00f5es: 22","OMN\u003cbr\u003eConex\u00f5es: 33","ABW\u003cbr\u003eConex\u00f5es: 14","SPM\u003cbr\u003eConex\u00f5es: 1","SLE\u003cbr\u003eConex\u00f5es: 14","TGO\u003cbr\u003eConex\u00f5es: 17","AFG\u003cbr\u003eConex\u00f5es: 11","IRN\u003cbr\u003eConex\u00f5es: 15","IRQ\u003cbr\u003eConex\u00f5es: 23","LBY\u003cbr\u003eConex\u00f5es: 24","MDV\u003cbr\u003eConex\u00f5es: 22","PAK\u003cbr\u003eConex\u00f5es: 125","TJK\u003cbr\u003eConex\u00f5es: 13","BTN\u003cbr\u003eConex\u00f5es: 6","SWZ\u003cbr\u003eConex\u00f5es: 8","BLZ\u003cbr\u003eConex\u00f5es: 17","COK\u003cbr\u003eConex\u00f5es: 7","MWI\u003cbr\u003eConex\u00f5es: 8","TLS\u003cbr\u003eConex\u00f5es: 7","LAO\u003cbr\u003eConex\u00f5es: 12","ATA\u003cbr\u003eConex\u00f5es: 4","VGB\u003cbr\u003eConex\u00f5es: 11","BES\u003cbr\u003eConex\u00f5es: 2","SLB\u003cbr\u003eConex\u00f5es: 8","NIU\u003cbr\u003eConex\u00f5es: 1","PCN\u003cbr\u003eConex\u00f5es: 1","TUV\u003cbr\u003eConex\u00f5es: 4","ATF\u003cbr\u003eConex\u00f5es: 2","ZWE\u003cbr\u003eConex\u00f5es: 15","MOZ\u003cbr\u003eConex\u00f5es: 15","GNB\u003cbr\u003eConex\u00f5es: 6","STP\u003cbr\u003eConex\u00f5es: 4","SOM\u003cbr\u003eConex\u00f5es: 13","NPL\u003cbr\u003eConex\u00f5es: 13","YEM\u003cbr\u003eConex\u00f5es: 9","LSO\u003cbr\u003eConex\u00f5es: 3","GMB\u003cbr\u003eConex\u00f5es: 9","GRD\u003cbr\u003eConex\u00f5es: 12","BDI\u003cbr\u003eConex\u00f5es: 8","DJI\u003cbr\u003eConex\u00f5es: 7","SSD\u003cbr\u003eConex\u00f5es: 11","SYR\u003cbr\u003eConex\u00f5es: 3","FLK\u003cbr\u003eConex\u00f5es: 2","KNA\u003cbr\u003eConex\u00f5es: 7","DMA\u003cbr\u003eConex\u00f5es: 10","CAF\u003cbr\u003eConex\u00f5es: 10","PSE\u003cbr\u003eConex\u00f5es: 7","UMI\u003cbr\u003eConex\u00f5es: 3","NER\u003cbr\u003eConex\u00f5es: 13","FSM\u003cbr\u003eConex\u00f5es: 5","CXR\u003cbr\u003eConex\u00f5es: 2","ASM\u003cbr\u003eConex\u00f5es: 3","ERI\u003cbr\u003eConex\u00f5es: 6","SDN\u003cbr\u003eConex\u00f5es: 11","ESH\u003cbr\u003eConex\u00f5es: 2","AIA\u003cbr\u003eConex\u00f5es: 5","MNP\u003cbr\u003eConex\u00f5es: 4","NFK\u003cbr\u003eConex\u00f5es: 2","PLW\u003cbr\u003eConex\u00f5es: 3","IOT\u003cbr\u003eConex\u00f5es: 1","SHN\u003cbr\u003eConex\u00f5es: 2","MSR\u003cbr\u003eConex\u00f5es: 1","TKL\u003cbr\u003eConex\u00f5es: 2"],"visible":false,"type":"scattergeo"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"projection":{"type":"equirectangular"},"showland":true,"landcolor":"lightgray","showocean":true,"oceancolor":"lightblue","showcountries":true,"countrycolor":"white"},"title":{"text":"Evolu\u00e7\u00e3o das Redes de Exporta\u00e7\u00e3o de M\u00e1scaras Cir\u00fargicas"},"height":600,"sliders":[{"active":0,"currentvalue":{"prefix":"Per\u00edodo: "},"steps":[{"args":[{"visible":[true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2015","method":"update"},{"args":[{"visible":[false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2016","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2017","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2018","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2019","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2020","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false,false,false,false,false]}],"label":"2021","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false,false,false,false,false]}],"label":"2022","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,false,false,false,false]}],"label":"2023","method":"update"},{"args":[{"visible":[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true]}],"label":"2024","method":"update"}]}]},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('cfe26ae9-70d6-44d7-bf3d-e77eb6d8b1c2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


## 4. Portugal


```python
def analisar_portugal_por_periodo(networks_periodos, coords):

    portugal_stats = {}
    
    for periodo, G in networks_periodos.items():
        print(f"\n{'='*60}")
        print(f"PORTUGAL - {periodo.upper()}")
        print(f"{'='*60}")
        
        #validar portugal em todos os periodos
        if 'PRT' not in G.nodes():
            print(f"Portugal não encontrado na rede do período {periodo}")
            continue
            
        # estatísticas básicas 
        grau_total = G.degree('PRT')
        grau_entrada = G.in_degree('PRT') 
        grau_saida = G.out_degree('PRT')   
        
        print(f"Estatísticas Básicas:")
        print(f"  - Grau Total: {grau_total}")
        print(f"  - Exportações (grau saída): {grau_saida}")
        print(f"  - Importações (grau entrada): {grau_entrada}")
        
        # nodos com lig dorectas
        exportadores_portugal = list(G.predecessors('PRT'))  
        importadores_portugal = list(G.successors('PRT'))    
        
        print(f"\nParceiros Comerciais:")
        print(f"  - Exporta para {len(importadores_portugal)} países: {importadores_portugal}")
        print(f"  - Importa de {len(exportadores_portugal)} países: {exportadores_portugal}")
        
        # centralidades

        betweenness = nx.betweenness_centrality(G)['PRT']
        closeness = nx.closeness_centrality(G)['PRT'] 
        eigenvector = nx.eigenvector_centrality(G)['PRT']
        pagerank = nx.pagerank(G)['PRT']
            
        print(f"\nMedidas de Centralidade:")
        print(f"  - Betweenness: {betweenness:.4f}")
        print(f"  - Closeness: {closeness:.4f}")
        print(f"  - Eigenvector: {eigenvector:.4f}")
        print(f"  - PageRank: {pagerank:.4f}")

        
        # ranking de Portugal em grau
        all_degrees = dict(G.degree())
        portugal_rank = sorted(all_degrees.values(), reverse=True).index(grau_total) + 1
        total_paises = len(G.nodes())
        
        print(f"\nPosição Relativa:")
        print(f"  - Ranking por grau: {portugal_rank}º de {total_paises} países")
        print(f"  - Percentil: {((total_paises - portugal_rank) / total_paises * 100):.1f}%")
        
        # guardar stats
        portugal_stats[periodo] = {
            'grau_total': grau_total,
            'grau_entrada': grau_entrada,
            'grau_saida': grau_saida,
            'exportadores': exportadores_portugal,
            'importadores': importadores_portugal,
            'ranking': portugal_rank,
            'total_paises': total_paises
        }
    
    return portugal_stats


def comparar_evolucao_portugal(portugal_stats):

    print(f"\n{'='*60}")
    print("EVOLUÇÃO DE PORTUGAL ENTRE PERÍODOS")
    print(f"{'='*60}")
    
    periodos = list(portugal_stats.keys())
    
    df_portugal = pd.DataFrame(portugal_stats).T
    print("\nTabela Comparativa:")
    display(df_portugal[['grau_total', 'grau_entrada', 'grau_saida', 'ranking', 'total_paises']])
    
    # gráficos de evolução
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Evolução de Portugal nas Redes de Exportação de Máscaras", fontsize=16)
    
    #grau total
    axes[0, 0].plot(periodos, [portugal_stats[p]['grau_total'] for p in periodos], 'o-', color='blue')
    axes[0, 0].set_title('Grau Total')
    axes[0, 0].set_ylabel('Número de Conexões')
    axes[0, 0].grid(True, alpha=0.3)
    
    # export vs impor
    exportacoes = [portugal_stats[p]['grau_saida'] for p in periodos]
    importacoes = [portugal_stats[p]['grau_entrada'] for p in periodos]
    
    x = range(len(periodos))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], exportacoes, width, label='Exportações', color='green')
    axes[0, 1].bar([i + width/2 for i in x], importacoes, width, label='Importações', color='orange')
    axes[0, 1].set_title('Exportações vs Importações')
    axes[0, 1].set_ylabel('Número de Conexões')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(periodos)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rank
    axes[1, 0].plot(periodos, [portugal_stats[p]['ranking'] for p in periodos], 'o-', color='red')
    axes[1, 0].set_title('Posição no Ranking')
    axes[1, 0].set_ylabel('Posição (menor = melhor)')
    axes[1, 0].invert_yaxis()  # Inverter para melhor visualização
    axes[1, 0].grid(True, alpha=0.3)
    
    # percentil
    percentis = [((portugal_stats[p]['total_paises'] - portugal_stats[p]['ranking']) / portugal_stats[p]['total_paises'] * 100) for p in periodos]
    axes[1, 1].plot(periodos, percentis, 'o-', color='purple')
    axes[1, 1].set_title('Percentil de Conectividade')
    axes[1, 1].set_ylabel('Percentil (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # análise às mudanças
    print("\nAnálise de Mudanças:")
    for i in range(1, len(periodos)):
        periodo_anterior = periodos[i-1]
        periodo_atual = periodos[i]
        
        delta_grau = portugal_stats[periodo_atual]['grau_total'] - portugal_stats[periodo_anterior]['grau_total']
        delta_ranking = portugal_stats[periodo_atual]['ranking'] - portugal_stats[periodo_anterior]['ranking']
        
        print(f"\n{periodo_anterior} → {periodo_atual}:")
        print(f"  - Mudança no grau total: {delta_grau:+d}")
        print(f"  - Mudança no ranking: {delta_ranking:+d} posições")
        
        # novas ligações
        novos_exp = set(portugal_stats[periodo_atual]['exportadores']) - set(portugal_stats[periodo_anterior]['exportadores'])
        novos_imp = set(portugal_stats[periodo_atual]['importadores']) - set(portugal_stats[periodo_anterior]['importadores'])
        
        if novos_exp:
            print(f"  - Novos países que exportam para Portugal: {list(novos_exp)}")
        if novos_imp:
            print(f"  - Novos países que importam de Portugal: {list(novos_imp)}")

# Executar análise completa
print("O Caso de Portugal")
print()

# análise estatística por período
portugal_stats = analisar_portugal_por_periodo(networks_periodos, coords)


# comparação evolutiva
comparar_evolucao_portugal(portugal_stats)
```

    O Caso de Portugal
    
    
    ============================================================
    PORTUGAL - PRÉ-PANDEMIA
    ============================================================
    Estatísticas Básicas:
      - Grau Total: 29
      - Exportações (grau saída): 15
      - Importações (grau entrada): 14
    
    Parceiros Comerciais:
      - Exporta para 15 países: ['AGO', 'CHE', 'CZE', 'DEU', 'DNK', 'ESP', 'FRA', 'GBR', 'IRL', 'ITA', 'MAR', 'NLD', 'NOR', 'SWE', 'TUN']
      - Importa de 14 países: ['BEL', 'CHN', 'DEU', 'ESP', 'FRA', 'GBR', 'HKG', 'IND', 'ITA', 'MAR', 'NLD', 'ROU', 'TUR', 'VNM']
    
    Medidas de Centralidade:
      - Betweenness: 0.0010
      - Closeness: 0.1977
      - Eigenvector: 0.1195
      - PageRank: 0.0072
    
    Posição Relativa:
      - Ranking por grau: 23º de 159 países
      - Percentil: 85.5%
    
    ============================================================
    PORTUGAL - DURANTE PANDEMIA
    ============================================================
    Estatísticas Básicas:
      - Grau Total: 36
      - Exportações (grau saída): 19
      - Importações (grau entrada): 17
    
    Parceiros Comerciais:
      - Exporta para 19 países: ['AGO', 'AUT', 'BEL', 'CHE', 'CPV', 'CZE', 'DEU', 'DNK', 'ESP', 'FRA', 'GBR', 'IRL', 'ITA', 'NLD', 'POL', 'ROU', 'SVK', 'SWE', 'USA']
      - Importa de 17 países: ['BEL', 'CHE', 'CHN', 'CZE', 'DEU', 'ESP', 'FRA', 'GBR', 'HKG', 'IND', 'ITA', 'MAR', 'NLD', 'ROU', 'SVN', 'TUR', 'USA']
    
    Medidas de Centralidade:
      - Betweenness: 0.0021
      - Closeness: 0.2391
      - Eigenvector: 0.1393
      - PageRank: 0.0091
    
    Posição Relativa:
      - Ranking por grau: 22º de 175 países
      - Percentil: 87.4%
    
    ============================================================
    PORTUGAL - PÓS-PANDEMIA
    ============================================================
    Estatísticas Básicas:
      - Grau Total: 40
      - Exportações (grau saída): 21
      - Importações (grau entrada): 19
    
    Parceiros Comerciais:
      - Exporta para 21 países: ['AGO', 'BEL', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'DZA', 'ESP', 'FRA', 'GBR', 'IRL', 'ITA', 'MAR', 'NLD', 'NOR', 'POL', 'ROU', 'SVK', 'SWE', 'USA']
      - Importa de 19 países: ['BEL', 'CHN', 'CZE', 'DEU', 'DNK', 'ESP', 'FRA', 'GBR', 'GRC', 'ITA', 'MAR', 'NLD', 'POL', 'ROU', 'SVN', 'SWE', 'TUR', 'UKR', 'USA']
    
    Medidas de Centralidade:
      - Betweenness: 0.0020
      - Closeness: 0.2396
      - Eigenvector: 0.1499
      - PageRank: 0.0091
    
    Posição Relativa:
      - Ranking por grau: 21º de 167 países
      - Percentil: 87.4%
    
    ============================================================
    EVOLUÇÃO DE PORTUGAL ENTRE PERÍODOS
    ============================================================
    
    Tabela Comparativa:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grau_total</th>
      <th>grau_entrada</th>
      <th>grau_saida</th>
      <th>ranking</th>
      <th>total_paises</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pré-Pandemia</th>
      <td>29</td>
      <td>14</td>
      <td>15</td>
      <td>23</td>
      <td>159</td>
    </tr>
    <tr>
      <th>Durante Pandemia</th>
      <td>36</td>
      <td>17</td>
      <td>19</td>
      <td>22</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Pós-Pandemia</th>
      <td>40</td>
      <td>19</td>
      <td>21</td>
      <td>21</td>
      <td>167</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_66_2.png)
    


    
    Análise de Mudanças:
    
    Pré-Pandemia → Durante Pandemia:
      - Mudança no grau total: +7
      - Mudança no ranking: -1 posições
      - Novos países que exportam para Portugal: ['USA', 'SVN', 'CZE', 'CHE']
      - Novos países que importam de Portugal: ['SVK', 'CPV', 'AUT', 'BEL', 'USA', 'POL', 'ROU']
    
    Durante Pandemia → Pós-Pandemia:
      - Mudança no grau total: +4
      - Mudança no ranking: -1 posições
      - Novos países que exportam para Portugal: ['GRC', 'UKR', 'DNK', 'SWE', 'POL']
      - Novos países que importam de Portugal: ['DZA', 'MAR', 'BGR', 'NOR']
    


```python

```
