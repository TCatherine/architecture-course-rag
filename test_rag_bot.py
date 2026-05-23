import pytest
from pathlib import Path
import numpy as np
from bot import build_qa_chain, load_config
from sklearn.metrics.pairwise import cosine_similarity 

# Набор тестовых кейсов: вопрос и список слов, которые ОБЯЗАТЕЛЬНО должны быть в ответе
TEST_CASES = [
    ("Кто такой Kael?", ["Кел", "Каррадайн", "хранитель"]),
    ("Как звали брата Kael?", ["Рован"]),
    ("Как звали отца Kael и Rovan?", ["Tomar", "Eric", "Karradine"]),
    ("Какими способами умирал Kael?", ["застреленный", "машиной", "пианино", "сосиской"]),
    ("О чем Karradines?", ["любовь", "Том", "Льора"]),
    ("Что означает имя Kael?", ["лидер", "долина"]),
    ("Кто такой Rovan?", ["Рован", "Этроса"]),
    ("Где родился Kael?", ["нет", "информации"]),
    ("Какого цвета Vanguard?", ["черного"]),
    ("Что такое Vanguard?", ["машина"]),
]

@pytest.fixture(scope="module")
def setup():
    config = load_config(Path("./config.yaml"))
    qa = build_qa_chain(Path("./chroma.db"), config)
    # Используем тот же эмбеддер, что и в боте
    embeddings = qa.retriever.vectorstore.embeddings
    return qa, embeddings

def test_rag_with_similarity(setup):
    qa, embeddings = setup
    
    for query, expected_answer in TEST_CASES:
        # 1. Получаем ответ
        result = qa.invoke({"query": query})
        answer = result["result"]
        expected = str(expected_answer)
        
        # 2. Получаем векторы ответов
        vec_answer = np.array(embeddings.embed_query(answer)).reshape(1, -1)
        vec_expected = np.array(embeddings.embed_query(expected)).reshape(1, -1)
        
        # 3. Считаем косинусное сходство (от 0 до 1)
        similarity = cosine_similarity(vec_answer, vec_expected)[0][0]
        score = round(similarity * 5, 1)  # переводим в шкалу 1-5
        
        print(f"\nВопрос: {query}")
        print(f"Ответ: {answer}")
        print(f"Оценка (сходство): {score}/5")
        
        # 4. Проверка (порог 3.0 из 5.0)
        assert score >= 3.0, f"Ответ слишком сильно отличается от эталона. Оценка: {score}"