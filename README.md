# Sistema de Recomendação de Filmes

## Objetivo do projeto

Este projeto implementa um sistema de recomendação de filmes baseado em conteúdo (content-based filtering), utilizando processamento de texto e similaridade de cosseno.

O material foi organizado para apoiar a escrita de um artigo técnico, permitindo:

- execução direta via script Python;
- acompanhamento interativo e didático via notebook.

## Como o projeto funciona

O pipeline de recomendação segue estas etapas:

1. Download do dataset via API do Kaggle (`kagglehub`).
2. Carregamento e seleção dos campos relevantes (`title`, `genres`, `keywords`, `cast`, `director`).
3. Pré-processamento textual (normalização e tokenização).
4. Stemming com `PorterStemmer`.
5. Vetorização com `CountVectorizer`.
6. Cálculo de similaridade com cosseno.
7. Retorno dos filmes mais similares a um título de referência.

## Bibliotecas utilizadas

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `kagglehub`
- `ipykernel` (suporte a execução em notebook)

As dependências estão declaradas em `pyproject.toml` e são gerenciadas com `uv`.

## Estrutura principal

- `main.py`: script principal para execução do recomendador.
- `notebook.ipynb`: versão interativa para estudo passo a passo do pipeline.

Origem do dataset: [Kaggle - Movies CSV](https://www.kaggle.com/datasets/harshshinde8/movies-csv).

## Como executar

### 1. Instalar dependências

```bash
uv sync
```

### 2. Executar o script Python

```bash
uv run main.py
```

Esse comando baixa o dataset via `kagglehub`, processa os dados e imprime recomendações no terminal.

### 3. Executar de forma interativa (notebook)

```bash
uv run jupyter notebook
```

Depois, abra `notebook.ipynb` para acompanhar cada etapa do algoritmo de forma interativa, com foco em análise e explicação para o artigo.

## Observações

- O título consultado no script deve existir exatamente no dataset.
- Se desejar testar outros filmes, altere a variável `title` em `main.py`.
