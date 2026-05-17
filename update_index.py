import os
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HASH_FILE = "indexed_files.json"

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_indexed_files():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_indexed_files(hashes):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=4)

def process_and_index_chroma(directory_path, persist_directory="./chroma.db"):
    start_time = time.time()
    stats = {"added_chunks": 0, "errors": [], "deleted_files": 0}
    # 1. Загрузка состояния
    indexed_data = get_indexed_files()
    current_files = {f: calculate_sha256(os.path.join(directory_path, f)) 
                     for f in os.listdir(directory_path) if f.endswith(".md")}
    
    # 2. Определение изменений
    files_to_process = [f for f, h in current_files.items() if indexed_data.get(f) != h]
    files_to_remove = [f for f in indexed_data if f not in current_files]

    if not files_to_process and not files_to_remove:
        logger.info("Изменений не найдено.")
        return

    # Инициализация БД
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 3. Удаление старых данных
    if files_to_remove:
        stats["deleted_files"] = len(files_to_remove)
        vector_db.delete(where={"source": {"$in": files_to_remove}})

    # 4. Обработка новых файлов
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for filename in files_to_process:
        try:
            logger.info(f"Обработка файла: {filename}")
            file_path = os.path.join(directory_path, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            vector_db.add_documents(chunks)
            stats["added_chunks"] += len(chunks)
        except Exception as e:
            stats["errors"].append(f"{filename}: {str(e)}")
            logger.error(f"Ошибка при обработке {filename}: {e}")

    # 5. Итоговый отчет
    end_time = time.time()
    duration = end_time - start_time
    
    # Получаем размер индекса (количество документов в базе)
    total_count = vector_db._collection.count()
    
    logger.info("\n" + "="*40)
    logger.info("ОТЧЕТ ОБ ИНДЕКСАЦИИ")
    logger.info(f"Время выполнения:      {duration:.2f} сек.")
    logger.info(f"Добавлено чанков:     {stats['added_chunks']}")
    logger.info(f"Удалено старых файлов: {stats['deleted_files']}")
    logger.info(f"Итого чанков в базе:   {total_count}")
    
    if stats["errors"]:
        logger.info(f"Ошибок при обработке:  {len(stats['errors'])}")
        for err in stats["errors"]:
            logger.info(f" ! {err}")
    else:
        logger.info("Ошибок:                нет")
    logger.info("="*40 + "\n")

if __name__ == "__main__":
    process_and_index_chroma('./knowledge_base_anon')
