# Wind Decision Framework

Framework computacional integrado de apoio à decisão para seleção de áreas aptas à
implantação de parques eólicos, combinando Análise Multicritério em Sistemas de
Informação Geográfica (SIG–AHP) e modelagem preditiva da velocidade do vento por
Redes Neurais Artificiais.

Este repositório corresponde ao **produto tecnológico** desenvolvido no âmbito de
uma dissertação de mestrado acadêmico, atendendo às diretrizes da CAPES para
produção técnica e tecnológica.

---

##  Visão Geral

O *Wind Decision Framework* foi concebido para apoiar o planejamento energético
territorial, integrando:

- avaliação espacial de adequabilidade técnica;
- restrições legais e ambientais;
- previsibilidade temporal do recurso eólico.

O produto permite responder, de forma integrada, às seguintes questões:

- **Onde instalar?** (aptidão territorial e restrições)
- **Com que confiabilidade temporal?** (previsão do vento em múltiplos horizontes)

Dessa forma, o framework transcende análises puramente cartográficas ou estatísticas,
articulando informação espacial e desempenho temporal do vento em um único
pipeline metodológico.

---

##  Componentes do Produto Tecnológico

O produto é estruturado de forma modular e orientada a pipeline, contemplando:

- Análise multicritério espacial em SIG (SIG–AHP)
- Construção do mapa de exclusão territorial
- Cálculo do índice contínuo de adequabilidade técnica
- Geração do Mapa Final de Aptidão
- Extração e preparação de dados ambientais e meteorológicos
- Previsão da velocidade do vento por RNAs (MLP, LSTM, GRU e TCN)
- Avaliação estatística e análise de robustez espacial
- Integração dos resultados para apoio à decisão

---

## Reprodutibilidade

O produto foi desenvolvido segundo princípios de **reprodutibilidade e transparência
metodológica**.

A reprodução dos resultados (ou a obtenção de resultados equivalentes) é viabilizada
por meio de:

- scripts organizados por módulos;
- definição explícita de entradas e saídas;
- padronização de formatos de dados;
- documentação do ambiente computacional;
- instruções detalhadas no manual técnico (`docs/manual_produto.pdf`).

Todos os dados utilizados no estudo de caso são provenientes de **bases públicas**
(ERA5, Google Earth Engine, MapBiomas, IBGE), e o repositório fornece instruções
para sua obtenção.

---

##  Aplicabilidade

O framework **não se restringe à Região Norte Fluminense**.

Ele pode ser aplicado a outras regiões mediante:

- definição da área de estudo;
- fornecimento de camadas territoriais compatíveis;
- ajuste dos critérios e pesos AHP;
- extração de séries temporais climáticas para os pontos candidatos.

Essa característica confere ao produto caráter **transferível e adaptável**, alinhado
às exigências de produtos tecnológicos estabelecidas pela CAPES.

---

##  Público-Alvo

- Pesquisadores em planejamento energético e territorial
- Órgãos públicos e agências de planejamento
- Profissionais do setor de energias renováveis
- Estudantes e programas de pós-graduação

---

##  Requisitos Computacionais

- Python ≥ 3.10
- Ambiente descrito em `environment.yml`
- Dependências listadas em `requirements.txt`

O framework pode ser executado tanto em ambiente local quanto em ambientes de
nuvem (por exemplo, notebooks).

---

##  Referência

Se este produto for utilizado em trabalhos acadêmicos, recomenda-se citar a
dissertação associada ao seu desenvolvimento.
