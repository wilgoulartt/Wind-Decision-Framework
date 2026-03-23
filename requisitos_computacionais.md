# Requisitos Computacionais (Produto Tecnológico)

Este documento descreve os requisitos mínimos para reproduzir e executar o produto tecnológico (SIG–AHP + RNAs) desenvolvido nesta dissertação.

---

## 1) Requisitos de Hardware

### Mínimo recomendado (execução local)
- **CPU:** 4 núcleos (Intel i5/Ryzen 5 ou equivalente)
- **RAM:** 16 GB
- **Armazenamento:** 10–20 GB livres (dados + cache + saídas)
- **Sistema:** Windows 10/11, Linux ou macOS

### Recomendado (treinamento mais rápido das RNAs)
- **CPU:** 8+ núcleos
- **RAM:** 32 GB
- **GPU (opcional, mas recomendado):** NVIDIA com suporte a CUDA (para acelerar treinamento)
- **Armazenamento:** 30 GB+ (se incluir datasets/rasters adicionais)

> Observação: o produto pode ser executado sem GPU, porém o treinamento das RNAs pode levar mais tempo.

---

## 2) Requisitos de Software

### 2.1 Python
- **Python:** 3.10 ou 3.11 (recomendado)
- **Gerenciador de ambiente:** `conda` (recomendado) ou `venv`

### 2.2 Bibliotecas (núcleo do produto)
As principais dependências incluem:
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- TensorFlow/Keras (modelagem preditiva)
- GeoPandas / Rasterio (quando aplicável ao processamento geoespacial local)
- Earth Engine API + geemap (quando a extração/clima for via GEE)

> A lista completa deve estar no arquivo `requirements.txt` ou `environment.yml` (ver Seção 4).

### 2.3 Google Earth Engine (quando aplicável)
Se o pipeline utilizar dados via GEE:
- Conta Google habilitada no **Google Earth Engine**
- Autenticação ativa no ambiente (Colab ou local)

---

## 3) Modos de Execução Suportados

O produto pode ser executado em três cenários:

### (A) Google Colab (recomendado para reprodução rápida)
- Vantagens: menos fricção de ambiente, fácil uso de GPU, integração com GEE.
- Recomendado quando o usuário não possui ambiente Python configurado.

### (B) Execução local (Windows/Linux/macOS)
- Vantagens: controle total de pastas e dados.
- Requer instalação do ambiente (Python + dependências).

### (C) Ambiente híbrido
- Pré-processamento SIG local + treinamento RNAs em Colab (ou vice-versa).
- Útil quando rasters/pesos estão localmente e o treinamento exige GPU.

---

## 4) Instalação do Ambiente

### Opção 1 — Conda (recomendado)
1. Instale o Miniconda ou Anaconda.
2. Crie o ambiente:

```bash
conda create -n produto_eolico python=3.10 -y
conda activate produto_eolico
