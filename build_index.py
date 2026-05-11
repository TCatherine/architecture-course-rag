import os
import logging
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_and_index_chroma(directory_path, chunk_size=500, chunk_overlap=50, persist_directory="./chroma.db"):
    logger.info(f"Начало процесса индексации папки: {directory_path}")

    # 1. Инициализация модели
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Модель BAAI/bge-m3 успешно загружена.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None

    # 2. Подготовка сплиттера
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []
    
    # 3. Чтение файлов
    files = [f for f in os.listdir(directory_path) if f.endswith(".md")]
    logger.info(f"Найдено файлов для обработки: {len(files)}")

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            chunks = text_splitter.split_documents(docs)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename
                chunk.metadata["chunk_id"] = i
                chunk.metadata["title"] = filename.replace(".md", "")
            
            all_docs.extend(chunks)
            logger.info(f"Файл {filename} разбит на {len(chunks)} чанков.")
        except Exception as e:
            logger.warning(f"Не удалось обработать файл {filename}: {e}")

    # 4. Создание базы
    if not all_docs:
        logger.error("Нет данных для индексации. Завершение работы.")
        return None

    logger.info(f"Всего подготовлено {len(all_docs)} чанков. Запуск генерации эмбеддингов...")
    
    try:
        vector_db = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Индексация завершена. База сохранена в: {persist_directory}")
    except Exception as e:
        logger.error(f"Критическая ошибка при записи в ChromaDB: {e}")
        return None
    
    return vector_db

if __name__ == "__main__":
    db = process_and_index_chroma('./knowledge_base_anon')
