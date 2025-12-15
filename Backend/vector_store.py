# --------------------------------------------------
# File: ~/RAG_Chatbot/Backend/vector_store.py
# Description: FAISS 기반 벡터 DB + 코사인 유사도 검색
# --------------------------------------------------

import faiss
import json
import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== 경로 설정 =====
BASE_DIR = os.path.join(os.path.expanduser("~"), "RAG_Chatbot")
DB_DIR = os.path.join(BASE_DIR, "faiss_db")
os.makedirs(DB_DIR, exist_ok=True)

FAISS_PATH = os.path.join(DB_DIR, "vector.index")
METADATA_PATH = os.path.join(DB_DIR, "metadata.json")
MODEL_NAME = "BAAI/bge-m3"

# ===== 전역 변수 =====
faiss_index = None
metadata = []
embedder = None


# ===== Embedding 모델 & FAISS 로드 =====
def load_faiss_into_memory():
    global faiss_index, metadata, embedder

    print("🔵 Loading embedding model on CPU...")
    embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    print("🟢 Embedding model loaded.")

    # Load FAISS index (IP = Inner Product → cosine possible)
    if os.path.exists(FAISS_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_PATH)
            print(f"🟢 FAISS index loaded. Total vectors: {faiss_index.ntotal}")
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            faiss_index = None
    else:
        faiss_index = None
        print("⚪ No FAISS index found. Starting fresh.")

    # Load metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata[:] = data if isinstance(data, list) else []
            print(f"🟢 Metadata loaded. Total chunks = {len(metadata)}")
        except:
            metadata[:] = []
            print("⚪ Metadata load failed. Starting empty.")
    else:
        metadata[:] = []
        print("⚪ No metadata found. Starting fresh.")


# ===== Metadata 저장 =====
def save_metadata():
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ===== chunk → 임베딩 문자열 변환 (전략 확장 지원) =====
def extract_text_for_embedding(chunk: dict) -> str:

    # 1) text가 있으면 최우선
    if "text" in chunk and isinstance(chunk["text"], str) and chunk["text"].strip():
        return chunk["text"]

    # 2) law 구조
    law_keys = ["chapter", "section", "article", "clause", "title"]
    if any(k in chunk for k in law_keys):
        parts = []
        for k in law_keys:
            if k in chunk and isinstance(chunk[k], str):
                parts.append(chunk[k])
        return " ".join(parts)

    # 3) category 구조
    cat_keys = ["title", "subtitle", "url"]
    if any(k in chunk for k in cat_keys):
        parts = []
        for k in cat_keys:
            if k in chunk and isinstance(chunk[k], str):
                parts.append(chunk[k])
        return " ".join(parts)

    # 4) 일반 record → 문자열 중 가장 긴 것 선택
    values = [v for v in chunk.values() if isinstance(v, str)]
    if values:
        return max(values, key=len)

    # 5) fallback
    return json.dumps(chunk, ensure_ascii=False)


# ===== 임베딩 생성 (코사인 지원을 위해 normalize) =====
def embed_texts(text_list):
    vecs = embedder.encode(text_list, convert_to_numpy=True, batch_size=16)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype("float32")


# ===== 벡터 / 메타데이터 저장 =====
def save_faiss(chunks, file_name: str):
    global faiss_index, metadata

    if not chunks:
        print(f"⚠ 저장할 청크 없음: {file_name}")
        return

    existing_hashes = {m.get("hash", "") for m in metadata}

    embedding_texts = []
    new_meta = []

    # CSV / 반복 데이터 중복 방지 (index + filename 포함)
    for idx, c in enumerate(chunks):
        embed_text = extract_text_for_embedding(c)
        raw_string = f"{file_name}-{idx}-{embed_text}"
        h = hashlib.md5(raw_string.encode("utf-8")).hexdigest()

        if h in existing_hashes:
            continue

        embedding_texts.append(embed_text)
        new_meta.append({
            "id": len(metadata) + len(new_meta),
            "file_name": file_name,
            **c,
            "hash": h
        })

    if not embedding_texts:
        print("⚪ 모든 청크가 중복 — 저장 생략")
        return

    vectors = embed_texts(embedding_texts)
    dim = vectors.shape[1]

    if faiss_index is None or faiss_index.ntotal == 0:
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss_index = index
    else:
        existing_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
        index = faiss.IndexFlatIP(dim)
        index.add(existing_vectors)
        index.add(vectors)
        faiss_index = index

    metadata.extend(new_meta)

    faiss.write_index(faiss_index, FAISS_PATH)
    save_metadata()

    print(f"🟢 저장 완료 — 파일: {file_name}, 새 청크: {len(new_meta)}, 전체: {faiss_index.ntotal}")


# ===== 검색 (코사인 기반) =====
def search_faiss(query, top_k=3, strategy_filter=None, file_name_filter=None):
    global metadata, faiss_index

    if faiss_index is None:
        raise RuntimeError("FAISS index not initialized!")

    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)
    q_vec = q_vec.astype("float32")

    D, I = faiss_index.search(q_vec, top_k * 3)

    results = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(metadata):
            chunk = metadata[idx]

            # strategy 필터
            if strategy_filter:
                if chunk.get("strategy") != strategy_filter:
                    continue

            # 추가된 문서(file_name) 필터
            if file_name_filter:
                if chunk.get("file_name") not in file_name_filter:
                    continue

            results.append({**chunk, "score": float(score)})

        if len(results) >= top_k:
            break

    return results



