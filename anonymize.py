#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("anon")


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка и валидация словаря
# ──────────────────────────────────────────────────────────────────────────────


def load_terms_map(path: Path) -> dict[str, str]:
    """
    Читает категоризированный JSON и возвращает плоский словарь замен.
    Ключи, начинающиеся с подчёркивания (например, '_meta'), игнорируются.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл словаря не найден: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON в {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Ожидался JSON-объект в корне {path}, получен {type(data).__name__}")

    flat: dict[str, str] = {}
    categories_loaded: list[str] = []

    for category, entries in data.items():
        if category.startswith("_"):  # _meta и прочие служебные секции
            continue
        if not isinstance(entries, dict):
            log.warning("Пропускаю секцию '%s': ожидался объект, получен %s",
                        category, type(entries).__name__)
            continue

        for src, tgt in entries.items():
            if not isinstance(src, str) or not isinstance(tgt, str):
                log.warning("Пропускаю некорректную пару в '%s': %r → %r", category, src, tgt)
                continue
            if not src.strip() or not tgt.strip():
                log.warning("Пропускаю пустую пару в '%s': %r → %r", category, src, tgt)
                continue
            if src in flat and flat[src] != tgt:
                raise RuntimeError(
                    f"Конфликт в {path}: '{src}' уже маппится в '{flat[src]}', "
                    f"но в секции '{category}' указано '{tgt}'."
                )
            flat[src] = tgt

        categories_loaded.append(f"{category} ({len(entries)})")

    log.info("Загружено %d терминов из %d категорий: %s",
             len(flat), len(categories_loaded), ", ".join(categories_loaded))

    if not flat:
        raise RuntimeError(f"Словарь пуст: {path}")

    return flat


# ──────────────────────────────────────────────────────────────────────────────
# Логика замены
# ──────────────────────────────────────────────────────────────────────────────


def preserve_case(source: str, target: str) -> str:
    """Применяет регистр исходного слова к замене.
    'DEAN' → 'KAEL', 'Dean' → 'Kael', 'dean' → 'kael'."""
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target[0].upper() + target[1:]
    return target.lower()


def build_replacer(flat_map: dict[str, str]):
    """Возвращает функцию(text)->text, выполняющую все замены за один проход."""
    lookup = {k.lower(): v for k, v in flat_map.items()}
    # Сортируем по длине убывания, чтобы 'Dean Winchester' побеждал 'Dean'.
    keys = sorted(flat_map.keys(), key=lambda s: -len(s))
    # Используем lookahead/lookbehind вместо \b — стабильнее для апострофов и юникода
    pattern_str = r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)"
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)

    def replace(text: str) -> str:
        def sub(m: re.Match) -> str:
            src = m.group(0)
            tgt = lookup[src.lower()]
            return preserve_case(src, tgt)

        return pattern.sub(sub, text)

    return replace


# ──────────────────────────────────────────────────────────────────────────────
# Работа с файлами
# ──────────────────────────────────────────────────────────────────────────────


def slugify(name: str) -> str:
    name = unquote(name).replace("_", " ").strip()
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "page"


SOURCE_LINE_RE = re.compile(
    r"^(\s*>\s*Source:\s*)(https?://\S+)\s*$", flags=re.IGNORECASE
)


def rewrite_source_line(line: str, replacer) -> str:
    """
    Преобразует строку источника, чтобы не светить supernatural.fandom.com:
        > Source: https://supernatural.fandom.com/wiki/Heaven
        →
        > Source: kb://Solareth
    """
    m = SOURCE_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return line
    prefix, url = m.group(1), m.group(2)
    parts = [p for p in urlparse(url).path.split("/") if p]
    raw_title = unquote(parts[-1] if parts else "page").replace("_", " ")
    anon_title = replacer(raw_title).replace(" ", "_")
    return f"{prefix}kb://{anon_title}\n"


def anonymize_file(in_path: Path, out_dir: Path, replacer) -> Path:
    """Анонимизирует один .md файл. Возвращает путь к новому файлу."""
    text = in_path.read_text(encoding="utf-8")

    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if SOURCE_LINE_RE.match(line.rstrip("\n")):
            out_lines.append(rewrite_source_line(line, replacer))
        else:
            out_lines.append(replacer(line))

    out_text = "".join(out_lines)

    # Новое имя файла — из первого H1 после анонимизации
    m = re.search(r"^#\s+(.+?)\s*$", out_text, flags=re.MULTILINE)
    new_basename = slugify(m.group(1)) if m else slugify(in_path.stem)
    new_path = out_dir / f"{new_basename}.md"

    counter = 1
    while new_path.exists():
        new_path = out_dir / f"{new_basename}_{counter}.md"
        counter += 1

    new_path.write_text(out_text, encoding="utf-8")
    return new_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Папка с исходными .md файлами.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Папка для анонимизированных .md файлов.")
    parser.add_argument("--terms-map", type=Path, default=Path("./terms_map.json"),
                        help="Путь к JSON-словарю замен. По умолчанию ./terms_map.json")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    flat_map = load_terms_map(args.terms_map)
    replacer = build_replacer(flat_map)

    files = sorted(args.input.glob("*.md"))
    if not files:
        log.warning("В %s не найдено .md файлов", args.input)
        return

    log.info("Анонимизирую %d файлов: %s → %s",
             len(files), args.input.resolve(), args.output.resolve())

    for in_path in files:
        new_path = anonymize_file(in_path, args.output, replacer)
        log.info("  %s  →  %s", in_path.name, new_path.name)

    log.info("=" * 60)
    log.info("Готово. Файлов: %d. Словарь замен: %s",
             len(files), args.terms_map.resolve())


if __name__ == "__main__":
    main()
