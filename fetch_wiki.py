#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kb")


# Fandom (и многие другие вики на Cloudflare) блокируют запросы с
# «не-браузерным» User-Agent с ошибкой 403. Поэтому представляемся как реальный браузер.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
}

# Селекторы, которые вырезаем до извлечения текста.
# Это типичный мусор fandom/MediaWiki, который не несёт фактического содержания.
JUNK_SELECTORS = [
    ".mw-editsection",          # кнопки [edit] рядом с заголовками
    ".reference",               # инлайн-сноски [1], [2]
    ".references",              # список сносок
    ".reference-list",
    ".navbox",                  # навигационные боксы внизу
    ".navigation-not-searchable",
    ".infobox",                 # классические инфобоксы
    ".portable-infobox",        # fandom-инфобоксы
    ".pi-data",
    ".toc",                     # оглавление
    ".thumb",                   # превью изображений + подписи
    ".thumbcaption",
    ".gallery",
    ".wikia-gallery",
    ".noprint",
    ".printfooter",
    ".catlinks",                # категории внизу страницы
    ".page-footer",
    ".article-categories",
    ".wds-tabs",
    ".mw-empty-elt",
    ".messagebox",
    ".ambox",                   # шаблоны "эту статью надо доработать"
    ".hatnote",                 # "Эта статья о ..., см. также ..."
    ".dablink",
    "style",
    "script",
    "figure",
    "aside",
    # Внутренние блоки fandom UI
    ".global-navigation",
    ".community-header-wrapper",
    ".page-header__categories",
]

# Названия секций, после которых обычно идёт только мусор (сноски, ссылки).
# Удаляем сам заголовок и всё до следующего h2 того же уровня.
SECTION_BLACKLIST = {
    # английские
    "references", "external links", "see also", "notes and references",
    "notes", "sources", "appearances", "behind the scenes", "bibliography",
    "further reading", "in other media", "trivia",
    # русские
    "источники", "ссылки", "примечания", "см. также", "литература",
    "появления", "за кулисами", "интересные факты", "ссылки на источники",
}


def slugify(name: str) -> str:
    """Sanitize строку под имя файла. Сохраняем кириллицу/латиницу/цифры."""
    name = unquote(name).replace("_", " ").strip()
    # Убираем всё, кроме букв, цифр, пробелов, дефиса
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "page"


def extract_title_from_url(url: str) -> str:
    """Извлекаем имя страницы из /wiki/<Title>."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if "wiki" in parts:
        idx = parts.index("wiki")
        if idx + 1 < len(parts):
            return unquote(parts[idx + 1])
    return parts[-1] if parts else "page"


def fetch_via_api(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Загружает контент статьи через MediaWiki API: /api.php?action=parse
    Возвращает (title, html). API почти не блокируется fandom-ом,
    в отличие от прямого скрейпа HTML.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    api_url = f"{base}/api.php"
    page_title = extract_title_from_url(url)

    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "text",
        "redirects": "1",
        "formatversion": "2",
    }
    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        info = data["error"].get("info", "unknown error")
        raise RuntimeError(f"MediaWiki API: {info}")

    parse = data.get("parse", {})
    html = parse.get("text", "")
    title = parse.get("title", unquote(page_title).replace("_", " "))
    if not html:
        raise RuntimeError("Пустой ответ от MediaWiki API")

    # API возвращает только тело статьи; обернём в div.mw-parser-output,
    # чтобы существующие селекторы чистки работали как для прямого HTML.
    wrapped = f'<div class="mw-parser-output">{html}</div>'
    return title, wrapped


def fetch_html(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Возвращает (title, html). Стратегия:
      1) Пробуем MediaWiki API — самый надёжный путь.
      2) Если API не сработал — прямой GET страницы с браузерным User-Agent.
    """
    try:
        return fetch_via_api(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("  API не сработал (%s), пробую прямой HTML…", exc)

    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return "", resp.text  # title="" — попросим извлечь из <h1>


def remove_blacklisted_sections(content) -> None:
    """
    Удаляем секции вроде 'References', 'See also' и всё их содержимое
    до следующего заголовка того же или более высокого уровня.
    Работает прямо в DOM `content` (BeautifulSoup tag).
    """
    for header in list(content.find_all(["h2", "h3"])):
        title = header.get_text(" ", strip=True).lower().strip(" :")
        if title in SECTION_BLACKLIST:
            # удаляем заголовок и всё, что идёт после него, пока не встретим
            # следующий заголовок того же или старшего уровня
            level = int(header.name[1])
            sibling = header.find_next_sibling()
            header.decompose()
            while sibling is not None:
                if sibling.name and sibling.name in ("h2", "h3", "h4"):
                    if int(sibling.name[1]) <= level:
                        break
                next_sibling = sibling.find_next_sibling()
                sibling.decompose()
                sibling = next_sibling


def clean_html_to_text(html: str, title_hint: str = "") -> tuple[str, str]:
    """
    Возвращает (title, plain_text). Plain text — markdown-подобный:
    заголовки разделов помечены '## ', абзацы — обычным текстом, списки — '- '.

    title_hint: если уже знаем title (например, из API) — используем его и
    не пытаемся искать <h1> (которого в API-ответе нет).
    """
    soup = BeautifulSoup(html, "lxml")

    if title_hint:
        title = title_hint
    else:
        # Заголовок статьи — из <h1> (если html взят прямым GET-ом)
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else "Untitled"
    # Убираем у заголовка типичные суффиксы "| Wookieepedia | Fandom"
    title = re.split(r"\s+[|—-]\s+", title)[0].strip()

    # Основной контейнер контента в MediaWiki — div.mw-parser-output
    content = soup.select_one(".mw-parser-output") or soup.body or soup

    # Чистим мусор
    for selector in JUNK_SELECTORS:
        for tag in content.select(selector):
            tag.decompose()

    # Убираем нерелевантные секции целиком
    remove_blacklisted_sections(content)

    # Собираем читаемый текст
    blocks: list[str] = []
    for element in content.find_all(["h2", "h3", "h4", "p", "li"]):
        # Пропустить, если уже удалён родитель (после decompose цикл может содержать сирот)
        if not element.parent:
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in ("h2", "h3", "h4"):
            level = "#" * (int(element.name[1]))  # h2 -> ##, h3 -> ###
            blocks.append(f"\n{level} {text}\n")
        elif element.name == "li":
            blocks.append(f"- {text}")
        else:  # <p>
            blocks.append(text)

    body = "\n\n".join(blocks)

    # Финальная чистка
    body = re.sub(r"\[\d+\]", "", body)          # остатки [1][2] от сносок
    body = re.sub(r"\[edit\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = body.strip()

    return title, body


def process_url(url: str, out_dir: Path, sleep: float = 1.0, min_chars: int = 300) -> Path | None:
    """Скачивает, чистит, сохраняет одну страницу. Возвращает путь или None."""
    try:
        log.info("Загружаю %s", url)
        api_title, html = fetch_html(url)
    except requests.RequestException as exc:
        log.error("  ошибка загрузки: %s", exc)
        return None

    try:
        title, body = clean_html_to_text(html, title_hint=api_title)
    except Exception as exc:  # noqa: BLE001
        log.error("  ошибка парсинга: %s", exc)
        return None

    if len(body) < min_chars:
        log.warning("  слишком мало текста (%d симв.) — пропускаю", len(body))
        return None

    slug = slugify(extract_title_from_url(url))
    out_path = out_dir / f"{slug}.md"

    # Записываем с шапкой: title + исходный URL (полезно как метаданные для RAG)
    front_matter = (
        f"# {title}\n\n"
        f"> Source: {url}\n\n"
        f"---\n\n"
    )
    out_path.write_text(front_matter + body + "\n", encoding="utf-8")
    log.info("  сохранено: %s  (%d симв.)", out_path.name, len(body))

    time.sleep(sleep)
    return out_path


def read_urls(urls_file: Path) -> list[str]:
    urls = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачать и очистить страницы с fandom/MediaWiki для RAG-базы.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("urls_file", type=Path, help="Текстовый файл со списком URL (один на строку).")
    parser.add_argument("--out", type=Path, default=Path("./knowledge_base"),
                        help="Папка для .md файлов (создастся, если нет). По умолчанию ./knowledge_base")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Пауза между запросами в секундах (вежливость к сайту). По умолчанию 1.0")
    parser.add_argument("--min-chars", type=int, default=300,
                        help="Минимальная длина статьи для сохранения. По умолчанию 300.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    urls = read_urls(args.urls_file)
    log.info("В очереди %d URL → папка %s", len(urls), args.out.resolve())

    saved = 0
    failed = 0
    for url in urls:
        result = process_url(url, args.out, sleep=args.sleep, min_chars=args.min_chars)
        if result:
            saved += 1
        else:
            failed += 1

    log.info("=" * 60)
    log.info("Готово. Сохранено: %d  |  пропущено/ошибок: %d  |  всего: %d", saved, failed, len(urls))
    log.info("База знаний: %s", args.out.resolve())


if __name__ == "__main__":
    main()
