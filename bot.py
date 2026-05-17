#!/usr/bin/env python3
"""
RAG-бот на LangChain RetrievalQA + локальной LLM (Ollama).

Все промпты и параметры лежат в prompts.yaml — код не нужно менять,
чтобы экспериментировать с формулировками или числом чанков.

Зависимости:
    pip install langchain langchain-chroma langchain-huggingface \\
                langchain-ollama sentence-transformers pyyaml

Локальная LLM (один раз):
    brew install ollama                  # macOS, или скачать с https://ollama.com
    ollama serve                          # обычно стартует автоматически
    ollama pull llama3.1:8b               # или другая модель из prompts.yaml

Запуск:
    python rag_bot.py --db ./chroma.db --config ./prompts.yaml
    python rag_bot.py --query "Кто такой Veynar?"            # одиночный запрос
    python rag_bot.py --show-sources                          # с источниками
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import re
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from pathlib import Path

import yaml
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag")

DB_PATH = Path("./chroma.db")
CONFIG_PATH = Path("./config.yaml")


async def run_telegram_bot(qa: RetrievalQA, token: str):
    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        await message.answer("Привет! Я RAG-бот. Задавай вопросы по базе.")

    @dp.message()
    async def handle_message(message: types.Message):
        # Используем sync_to_async или просто вызываем, так как Ollama быстрая
        log.info("Запрос: %s", message.text)
        result = qa.invoke({"query": message.text})
        await message.answer(result["result"])

    log.info("Запуск Telegram-бота...")
    await dp.start_polling(bot)


def load_config(path: Path) -> dict:
    if not path.exists():
        log.error("Файл конфига не найден: %s", path)
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("system", "user"):
        if key not in cfg or not cfg[key]:
            log.error("В %s отсутствует обязательный ключ '%s'", path, key)
            sys.exit(1)
    for var in ("{context}", "{question}"):
        if var not in cfg["user"]:
            log.error("В user-prompt не найден плейсхолдер %s", var)
            sys.exit(1)
    log.info("Файл конфигурации корректный")
    return cfg



def is_safe_chunk(text: str) -> bool:
    """
    Простейшая проверка на вредоносный контент.
    """
    forbidden_patterns = [
        r"Ignore all instructions", r"пароль", r"инструкция по краже"
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False        
    return True


def build_qa_chain(db_path: Path, config: dict) -> RetrievalQA:
    """
    Собирает готовую RetrievalQA-цепочку.

    Что делает RetrievalQA под капотом (для понимания):
      1. retriever.invoke(query)               # query → embedder → Chroma → top-k Document
      2. для каждого Document применяется document_prompt и они склеиваются
         через document_separator → строка {context}
      3. подставляется в {context} и {question} нашего qa_prompt
      4. отправляется в llm
      5. ответ возвращается + source_documents
    """

    log.info("Загружаю эмбеддер: BAAI/bge-m3")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    log.info("Открываю Chroma: %s", db_path.resolve())
    if not db_path.exists():
        log.error("База Chroma не найдена. Сначала запустите скрипт индексации.")
        sys.exit(1)
    vectordb = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )
    try:
        count = vectordb._collection.count()
        log.info("В индексе чанков: %d", count)
        if count == 0:
            log.error("Индекс пуст. Сначала запустите индексацию.")
            sys.exit(1)
    except Exception:  # noqa: BLE001
        pass

    retr_cfg = config.get("retrieval", {})
    retriever = vectordb.as_retriever(
        search_type=retr_cfg.get("search_type", "similarity"),
        search_kwargs={"k": retr_cfg.get("top_k", 4)},
    )
    log.info("Retriever: search_type=%s, k=%d",
             retr_cfg.get("search_type", "similarity"),
             retr_cfg.get("top_k", 4))

    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "llama3.1:8b")
    temperature = llm_cfg.get("temperature", 0.2)
    log.info("Подключаю Ollama: %s (T=%.2f)", model_name, temperature)
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=llm_cfg.get("num_ctx", 8192),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )

    examples_promts = [(k, e[k]) for e in config["examples"] for k in e.keys()]
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", config["system"])] +
        examples_promts +
        [("human", config["user"]),
    ])

    doc_template = config.get("document", "{page_content}")
    # Определяем, какие переменные нужны — RetrievalQA проверяет это строго.
    doc_input_vars = [v for v in ("page_content", "source", "title", "chunk_id")
                      if "{" + v + "}" in doc_template]
    if "page_content" not in doc_input_vars:
        doc_input_vars.append("page_content")
    doc_prompt = PromptTemplate(
        template=doc_template,
        input_variables=doc_input_vars,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                # все top-k чанков склеены в один промпт
        retriever=retriever,
        return_source_documents=True,      # вернуть и список Document
        chain_type_kwargs={
            "prompt": qa_prompt,
            "document_prompt": doc_prompt,
            "document_separator": config.get("document_separator", "\n\n"),
        },
    )

    return qa


def print_sources(docs) -> None:
    print("\n" + "─" * 60)
    print("Источники, использованные для ответа:")
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        line = f"  [{i}] {meta.get('source', '?')}"
        if "section" in meta:
            line += f"  · {meta['section']}"
        if "chunk_id" in meta:
            line += f"  · chunk #{meta['chunk_id']}"
        print(line)


def sanitize_input(question: str) -> str:
    injections = ["ignore all instructions", "забудь все инструкции", "ты теперь хакер", "пароль"]
    for pattern in injections:
        if pattern in question.lower():
            return False
    return True


def ask(qa: RetrievalQA, question: str, show_sources: bool = True) -> None:
    if sanitize_input(question) is False:
        print(f"\n\033[1;32mИзвините, я не могу выполнить этот запрос.\033[0m")
        return
        
    result = qa.invoke({"query": question})
    safe_docs = result if is_safe_chunk(result['result']) else {'result': "", 'source_documents': None}
    print(f"\n\033[1;32m{safe_docs['result']}\033[0m")
    if show_sources:
        print_sources(safe_docs["source_documents"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "tg"], default="cli")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    qa = build_qa_chain(DB_PATH, config)

    print("\nRAG-бот готов. Задавайте вопросы. 'exit' / Ctrl-D — выход.\n")

    if args.mode == "tg":
        token = config.get("telegram_token")
        if not token:
            log.error("В конфиге не найден telegram_token")
            sys.exit(1)
        asyncio.run(run_telegram_bot(qa, token))
    else:
        while True:
            try:
                question = input("\033[1;36m? \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"exit", "quit", "выход"}:
                break
            try:
                ask(qa, question)
            except Exception as exc:  # noqa: BLE001
                log.error("Ошибка при обработке запроса: %s", exc)


if __name__ == "__main__":
    main()
