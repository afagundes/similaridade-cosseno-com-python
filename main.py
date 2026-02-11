import pandas as pd
import numpy as np
import re
import logging

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    shape = data.shape

    logger.info("Dados carregados de %s com %s linhas x %s colunas", filepath, shape[0], shape[1])

    data = data[['title', 'genres', 'keywords', 'cast', 'director']]

    logger.info("Removendo linhas com valores vazios")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    shape = data.shape

    logger.info("Após remover linhas com valores vazios: %s linhas x %s colunas", shape[0], shape[1])

    return data


def normalize(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pré-processando os dados")

    data_tags = pd.DataFrame({
        "tags": (data["genres"] + " " + data["keywords"] + " " + data["cast"] + " " + data["director"])
    })

    data_tags["tags"] = data_tags["tags"].apply(normalize)

    return data_tags


def stem(text: list[str]) -> str:
    stemmer = PorterStemmer()

    words = []

    for w in text:
        words.append(stemmer.stem(w))

    return ' '.join(words)


def apply_stemmer(data_tags: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aplicando PorterStemmer")
    data_tags["tags"] = data_tags["tags"].apply(stem)
    return data_tags


def vectorize(data_tags: pd.DataFrame) -> np.ndarray:
    logger.info("Criando vetores a partir dos dados")

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data_tags['tags']).toarray()

    logger.info(
        "Matriz criada com %s vetores e %s dimensões cada",
        len(vectors),
        len(cv.get_feature_names_out())
    )

    return vectors


def calc_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    logger.info("Calculando similaridades de cosseno")
    similarities = cosine_similarity(vectors)

    logger.info(
        "Matriz criada com %s vetores e %s dimensões cada",
        len(similarities),
        len(similarities[0])
    )

    return similarities


def recommend(title: str, data: pd.DataFrame, similarities: np.ndarray) -> pd.DataFrame:
    matches = data[data['title'] == title]

    if matches.empty:
        raise ValueError(f"Título '{title}' não foi encontrado no conjunto de dados.")

    index = matches.index[0]

    distances = similarities[index]
    distances = list(enumerate(distances))

    distances.sort(key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in distances[1:6]:
        recommendations.append({
            "title": data.iloc[i[0]]["title"],
            "genres": data.iloc[i[0]]["genres"],
            "cast": data.iloc[i[0]]["cast"],
            "director": data.iloc[i[0]]["director"],
            "similarity": i[1]
        })

    return pd.DataFrame(recommendations)


def main():
    data = load_data("data/movies.csv")
    data_tags = preprocess_data(data)
    data_tags = apply_stemmer(data_tags)

    vectors = vectorize(data_tags)
    similarities = calc_cosine_similarity(vectors)

    title = "Mission: Impossible"
    # title = "The Lord of the Rings: The Fellowship of the Ring"

    recommendation = recommend(
        title=title,
        data=data,
        similarities=similarities
    )

    logger.info("Recomendações com base no filme '%s':\n\n%s", title, recommendation.to_string(index=False))



if __name__ == "__main__":
    main()
