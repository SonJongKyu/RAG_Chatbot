# --------------------------------------------------
# File: file_handler.py (FINAL FIXED VERSION)
# Description: 텍스트 추출 및 텍스트 청크 생성 기능
# --------------------------------------------------

import fitz
import re
import csv
import os
import json
from typing import List, Dict

BASE_DIR = os.path.join(os.path.expanduser("~"), "RAG_Chatbot")
CONFIG_PATH = os.path.join(BASE_DIR, "chunk_config.json")

DEFAULT_CONFIG = {
    "default": {"strategy": "regular", "chunk_size": 800, "overlap": 80},
    "pdf": {},
    "csv": {}
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG

#  ===== PDF Reader — 줄바꿈 유지 + 페이지 텍스트를 리스트로 반환 =====
def pdf_to_text_with_page(pdf_path: str, file_name: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # 🔥 줄바꿈은 유지해야 category / law 파서 사용 가능
        text = page.get_text("text")
        text = text.replace("\r", "").strip()

        pages.append({
            "page_no": page.number + 1,
            "text": text,
            "file_name": file_name
        })
    doc.close()
    return pages



#  ===== CSV Reader =====
def csv_to_text(file_path: str) -> str:
    rows = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(",".join(row))
    return "\n".join(rows)

#  ===== CATEGORY PARSER — category.pdf 전용 파서 =====
def parse_category_structure(raw_text: str) -> List[Dict]:
    import re

    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    chunks = []

    # 두 줄 패턴만 사용
    title_mark_re = re.compile(r"^\d+\.$")        # "1."
    subtitle_mark_re = re.compile(r"^[A-Z]\.$")   # "A."
    item_mark_re = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.$", re.IGNORECASE)
    url_re = re.compile(r"\((https?://[^\)]+)\)")

    title = None
    subtitle = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # 1) Title = "1." + 다음 줄 텍스트
        if title_mark_re.match(line):
            if i + 1 < len(lines):
                title = lines[i + 1].strip()
                subtitle = None
            i += 2
            continue

        # 2) Subtitle = "A." + 다음 줄 텍스트
        if subtitle_mark_re.match(line):
            if i + 1 < len(lines):
                subtitle = lines[i + 1].strip()
            i += 2
            continue

        # 3) Item = "i." + 다음 줄 텍스트 + (URL)
        if item_mark_re.match(line):
            item_text = None
            item_url = ""

            # 다음 줄 = 텍스트
            if i + 1 < len(lines):
                item_text = lines[i + 1].strip()
                i += 2
            else:
                i += 1
                continue

            # 그 다음 줄 = URL인지 검사
            if i < len(lines):
                m = url_re.search(lines[i])
                if m:
                    item_url = m.group(1)
                    i += 1

            # 저장
            if item_text:
                chunks.append({
                    "strategy": "category",
                    "title": title,
                    "subtitle": subtitle,
                    "text": item_text,
                    "url": item_url
                })

            continue

        i += 1

    return chunks

# ===== LAW PARSER — 전통시장법 / 시행령 / 시행규칙 전용 =====
def parse_law_pdf_text(text: str) -> List[Dict]:
    """
    전통시장법 / 시행령 / 시행규칙 등 법령 PDF 파싱 전용 함수
    - 한 줄에 조문이 여러 개 붙어 있어도 처리 가능
    - 제N장, 제N절, 제N조, (조문명), ①②③ 등 처리
    """

    chunks = []

    # 정규식
    chapter_re = re.compile(r"(제\d+장\s*[^\s]*)")
    section_re = re.compile(r"(제\d+절\s*[^\s]*)")
    article_re = re.compile(r"(제\d+조)\s*\((.*?)\)")
    clause_re = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩])")

    # 현재 상태
    current_chapter = "-"
    current_section = "-"

    # 텍스트 전체에서 모두 탐색
    chapters = list(chapter_re.finditer(text))
    sections = list(section_re.finditer(text))
    articles = list(article_re.finditer(text))

    # chapter & section 위치 인덱싱
    chapter_positions = {m.start(): m.group(1) for m in chapters}
    section_positions = {m.start(): m.group(1) for m in sections}

    # 각 조문(article) 순회
    for idx, art in enumerate(articles):
        article = art.group(1)
        title = art.group(2).strip()
        start = art.end()

        # 다음 조문 시작 지점
        end = articles[idx+1].start() if idx + 1 < len(articles) else len(text)

        body = text[start:end].strip()

        # 현재 조문 앞에 chapter/section이 있는지 확인
        for pos in sorted(chapter_positions.keys()):
            if pos < art.start():
                current_chapter = chapter_positions[pos]
            else:
                break

        for pos in sorted(section_positions.keys()):
            if pos < art.start():
                current_section = section_positions[pos]
            else:
                break

        # 항 분리
        clauses = clause_re.split(body)

        # 항이 없는 조문
        if len(clauses) <= 1:
            chunks.append({
                "strategy": "law",
                "chapter": current_chapter,
                "section": current_section,
                "article": article,
                "title": title,
                "clause": "-",
                "text": body
            })
            continue

        # 항이 있는 조문
        for i in range(1, len(clauses), 2):
            clause_no = clauses[i]        # ①
            clause_text = clauses[i+1].strip()  # 내용

            chunks.append({
                "strategy": "law",
                "chapter": current_chapter,
                "section": current_section,
                "article": article,
                "title": title,
                "clause": clause_no,
                "text": clause_text
            })

    return chunks

# ===== REGULAR CHUNK =====
def chunk_regular(text: str, cfg) -> List[Dict]:
    chunk_size = cfg.get("chunk_size", 800)
    overlap = cfg.get("overlap", 80)
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        blocks.append({"text": text[start:end]})
        start += max(chunk_size - overlap, 1)
    return blocks

#  =====  COLUMN RECORD (CSV) =====
def chunk_column_record(text: str, cfg) -> List[Dict]:
    mapping = cfg.get("mapping", {})
    rows = [line.split(",") for line in text.splitlines() if line.strip()]

    out = []
    for row in rows:
        obj = {k: (row[idx] if idx < len(row) else None)
               for k, idx in mapping.items()}
        obj["strategy"] = "csv"
        obj["page_no"] = "-"
        out.append(obj)
    return out

# ===== PAGE STRATEGY (onnurigift) =====
def chunk_page(text: str) -> List[Dict]:
    return [{"strategy": "page", "text": text}]

# ===== APPLY STRATEGY =====
def get_chunk_strategy(file_name: str):
    cfg = load_config()
    ext = "pdf" if file_name.lower().endswith(".pdf") else "csv"
    return cfg.get(ext, {}).get(file_name, cfg.get("default", {}))


def apply_chunk_strategy(raw_text: str, file_name: str) -> List[Dict]:
    cfg = get_chunk_strategy(file_name)
    strategy = cfg.get("strategy", "regular")

    if strategy == "law":
        return parse_law_pdf_text(raw_text)

    elif strategy == "category":
        return parse_category_structure(raw_text)

    elif strategy == "column_record":
        return chunk_column_record(raw_text, cfg)

    elif strategy == "page":
        return chunk_page(raw_text)

    else:
        return chunk_regular(raw_text, cfg)

def chunk_text_dynamic(text: str, file_name: str) -> List[Dict]:
    return apply_chunk_strategy(text, file_name)

chunk_text = chunk_text_dynamic
