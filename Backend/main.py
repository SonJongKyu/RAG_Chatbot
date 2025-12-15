# --------------------------------------------------
# File: ~/RAG_Chatbot/Backend/main.py
# Description: FastAPI 기반 RAG 서버 메인 Entry Point
# --------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from datetime import datetime
import uuid
import threading
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from file_handler import pdf_to_text_with_page, csv_to_text, apply_chunk_strategy
from vector_store import save_faiss, search_faiss, load_faiss_into_memory
import vector_store

# ===== 하이브리드 리랭킹 =====
from ranking import hybrid_rank

# ===== 의도 파악 =====
from intent_classifier import classify_intent

# ===== 서버 시작 시 FAISS 로드 =====
load_faiss_into_memory()
app = FastAPI()

# ===== CORS 설정 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 디렉터리 =====
BASE_DIR = os.path.join(os.path.expanduser("~"), "RAG_Chatbot")
UPLOAD_DIR = os.path.join(BASE_DIR, "input")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_history_sessions")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# ===== 세션 저장 =====
def get_session_file(session_id: str):
    return os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")

def save_chat_history(question=None, answer=None, source=None, session_id=None, system_message=None):
    record = {"timestamp": datetime.now().isoformat()}
    if system_message: record["system_message"] = system_message
    if question: record["question"] = question
    if answer: record["answer"] = answer
    if source: record["source"] = source

    session_file = get_session_file(session_id)
    history = []
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            pass

    history.append(record)
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ===== 모델 =====
class Question(BaseModel):
    question: str

class SystemMessage(BaseModel):
    message: str
    session_id: str

@app.get("/")
def read_root():
    return {"status": "ok"}

# ===== 파일 업로드 + 임베딩 =====
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = []
        if file.filename.lower().endswith(".pdf"):
            pages = pdf_to_text_with_page(file_path, file.filename)
            for p in pages:
                for c in apply_chunk_strategy(p["text"], file.filename):
                    # ★ 수정 ① — strategy 필드를 chunk에 추가 ★
                    chunks.append({"page_no": p["page_no"], "strategy": c.get("strategy"), **c})
        else:
            text = csv_to_text(file_path)
            for c in apply_chunk_strategy(text, file.filename):
                # ★ 수정 ① — strategy 필드를 chunk에 추가 ★
                chunks.append({"page_no": "-", "strategy": c.get("strategy"), **c})

        save_faiss(chunks, file_name=file.filename)
        return {"filename": file.filename, "status": "업로드 + 임베딩 완료", "chunks": len(chunks)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

#  ===== WATCHER (input 폴더 감시) =====
class FileWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory: return
        _, ext = os.path.splitext(event.src_path)
        if ext.lower() not in [".pdf", ".csv"]: return

        print(f"[WATCHER] 새 파일 감지: {event.src_path}")
        time.sleep(0.5)

        try:
            filename = os.path.basename(event.src_path)
            chunks = []

            if filename.lower().endswith(".pdf"):
                pages = pdf_to_text_with_page(event.src_path, filename)
                for p in pages:
                    for c in apply_chunk_strategy(p["text"], filename):
                        # ★ 수정 ② — strategy 필드 추가 ★
                        chunks.append({"page_no": p["page_no"], "strategy": c.get("strategy"), **c})
            else:
                text = csv_to_text(event.src_path)
                for c in apply_chunk_strategy(text, filename):
                    # ★ 수정 ② — strategy 필드 추가 ★
                    chunks.append({"page_no": "-", "strategy": c.get("strategy"), **c})

            save_faiss(chunks, file_name=filename)
            print(f"[WATCHER] {filename} 자동 임베딩 완료 (chunks={len(chunks)})")
        except Exception as e:
            print(f"[WATCHER] 자동 임베딩 오류: {e}")

def start_watcher():
    observer = Observer()
    observer.schedule(FileWatcher(), UPLOAD_DIR, recursive=False)
    observer.start()
    print(f"[WATCHER] input 폴더 감시 시작: {UPLOAD_DIR}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

threading.Thread(target=start_watcher, daemon=True).start()

#  ===== 세션 API =====
@app.post("/new_chat_session")
def new_chat_session():
    session_id = str(uuid.uuid4())
    with open(get_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump([], f)
    return {"session_id": session_id}

@app.get("/list_chat_sessions")
def list_chat_sessions():
    sessions = []
    for f in os.listdir(CHAT_HISTORY_DIR):
        if f.endswith(".json"):
            sessions.append(f.replace(".json", ""))
    return {"sessions": sessions}

@app.get("/get_chat_history/{session_id}")
def get_chat_history(session_id: str):
    session_file = get_session_file(session_id)
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return {"session_id": session_id, "history": json.load(f)}
        except:
            return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": []}

@app.delete("/delete_chat_session/{session_id}")
def delete_chat_session(session_id: str):
    if not session_id or session_id == "undefined":
        return {"status": "skip", "reason": "invalid session_id"}
    session_file = get_session_file(session_id)
    if os.path.exists(session_file):
        os.remove(session_file)
        return {"status": "삭제 완료", "session_id": session_id}
    return {"status": "해당 세션 없음", "session_id": session_id}

@app.post("/save_system_message")
def save_system_message(data: SystemMessage):
    message_to_save = data.message
    session_id = data.session_id

    record = {"timestamp": datetime.now().isoformat()}

    QUESTION_MESSAGES = [
        "온누리 상품권 관련 업무 조회하겠습니다.",
        "온누리 상품권 가맹점 정보 조회하겠습니다."
    ]

    if message_to_save in QUESTION_MESSAGES:
        record["question"] = message_to_save
    else:
        record["system_message"] = message_to_save

        if message_to_save == "저는 온누리 상품권 관련 챗봇입니다. 무엇을 도와드릴까요?":
            record["buttons"] = ["업무조회", "가맹점조회"]

        elif message_to_save == "다음으로 무엇을 알고 싶으신가요?":
            record["buttons"] = [
                "온누리상품권", "발행 종류", "사용처", "소득공제",
                "유효기간", "가맹점 가입 방법", "가맹점 혜택"
            ]

    session_file = get_session_file(session_id)
    history = []
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            pass

    history.append(record)
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {"status": "ok"}

# ===== ANSWER EXTRACT =====
def extract_answer(chunk: dict):
    if "text" in chunk and chunk["text"]:
        return chunk["text"]

    lines = []
    for k, v in chunk.items():
        if k not in ["id", "page_no", "file_name", "hash"]:
            lines.append(f"{k}: {v}")

    return "\n".join(lines)

# ===== 우선 검색 로직 =====
CSV_FIELDS = ["가맹점코드", "가맹점명", "가맹주성함", "가맹주번호", "사업자번호"]

def search_csv_exact_or_partial(query: str):
    query = query.strip()

    exact = [
        m for m in vector_store.metadata
        if any(m.get(k) == query for k in CSV_FIELDS)
    ]
    if exact:
        return exact

    partial = [
        m for m in vector_store.metadata
        if any(query in m.get(k, "") for k in CSV_FIELDS)
    ]
    if partial:
        return partial

    return None

# ===== RAG QUERY =====
@app.post("/rag_query")
def rag_query(
    q: Question,
    session_id: str = Query(None),
    forced_intent: str = Query(None)
):
    try:

        # 세션 초기화
        if not session_id or session_id == "undefined":
            session_id = str(uuid.uuid4())
            with open(get_session_file(session_id), "w", encoding="utf-8") as f:
                json.dump([], f)

        # intent 결정 (SBERT Router)
        if forced_intent is not None:
            intent = forced_intent
        else:
            intent_result = classify_intent(q.question)
            intent = intent_result["intent"]

        # AMBIGUOUS → clarify
        if forced_intent is None and intent == "AMBIGUOUS":

            # 사용자 질문 저장
            save_chat_history(
                question=q.question,
                session_id=session_id
            )

            # system message 저장
            save_chat_history(
                session_id=session_id,
                system_message="어떤 정보를 원하시나요?"
            )

            # 버튼 추가
            session_file = get_session_file(session_id)
            history = []
            if os.path.exists(session_file):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except:
                    pass

            if history:
                history[-1]["buttons"] = [
                    "가맹점 조회",
                    "업무 페이지",
                    "업무 정보",
                    "관련 법령"
                ]
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)

            return {
                "session_id": session_id,
                "type": "clarify",
                "message": "어떤 정보를 원하시나요?",
                "options": [
                    {"label": "가맹점 조회", "intent": "MERCHANT_DATA"},
                    {"label": "업무 페이지", "intent": "SYSTEM_MENU"},
                    {"label": "업무 정보", "intent": "ONNURI_KNOWLEDGE"},
                    {"label": "관련 법령", "intent": "LAW"}
                ]
            }

        # 1️⃣ MERCHANT_DATA → information.csv
        if intent == "MERCHANT_DATA":
            csv_hits = search_csv_exact_or_partial(q.question)
            if csv_hits:
                best = csv_hits[0]
                answer = extract_answer(best)
                save_chat_history(q.question, answer, best["file_name"], session_id)
                return {
                    "session_id": session_id,
                    "answer": answer,
                    "source": best["file_name"],
                    "matches": csv_hits
                }
            else:
                return {
                    "session_id": session_id,
                    "answer": "해당 가맹점 정보를 찾을 수 없습니다."
                }

        # 2️⃣ SYSTEM_MENU → category.pdf
        if intent == "SYSTEM_MENU":
            results = search_faiss(
                q.question,
                top_k=3,
                strategy_filter="category"
            )
            if results:
                best = results[0]
                answer = f"{best['text']}\n{best.get('url','')}"
                save_chat_history(q.question, answer, best["file_name"], session_id)
                return {
                    "session_id": session_id,
                    "answer": answer,
                    "source": best["file_name"]
                }

        # 3️⃣ ONNURI_KNOWLEDGE → onnurigift.pdf
        if intent == "ONNURI_KNOWLEDGE":
            results = search_faiss(
                q.question,
                top_k=8,
                file_name_filter=["onnurigift.pdf"]
            )

            if results:
                ranked = hybrid_rank(
                    q.question,
                    results,
                    vector_store.embedder
                )

                top_contexts = ranked[:3]
                context_text = "\n\n".join(
                    [c["text"] for c in top_contexts if c.get("text")]
                )

                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(
                    model="timHan/llama3korean8B4QKM:latest",
                    base_url="http://127.0.0.1:11434",
                    max_tokens=300
                )

                prompt = f"""
아래 문서 내용을 바탕으로 질문에 대해 정확하게 답변하세요.
문서에 없는 정보는 절대 추가하지 마세요.
중요 핵심만 2줄 이하로 요약
불필요한 설명, 부가 정보 금지
답변은 짧고 명확하게
요약 답변 앞에 어떤 설명도 쓰지마

질문: {q.question}
문서 내용: {context_text}

요약 답변:
"""
                response = llm.generate([prompt])
                answer = response.generations[0][0].text.strip()

                save_chat_history(
                    q.question,
                    answer,
                    f"{top_contexts[0]['file_name']} | {top_contexts[0].get('page_no','-')}",
                    session_id
                )

                return {
                    "session_id": session_id,
                    "answer": answer,
                    "source": f"{top_contexts[0]['file_name']} | {top_contexts[0].get('page_no','-')}"
                }

        # 4️⃣ LAW → 법령 3종
        if intent == "LAW":
            LAW_FILES = [
                "전통시장법.pdf",
                "전통시장법_시행령.pdf",
                "전통시장법_시행규칙.pdf"
            ]

            results = search_faiss(
                q.question,
                top_k=8,
                strategy_filter="law",
                file_name_filter=LAW_FILES
            )

            if results:
                ranked = hybrid_rank(
                    q.question,
                    results,
                    vector_store.embedder
                )

                top_contexts = ranked[:3]
                context_text = "\n\n".join(
                    [c["text"] for c in top_contexts if c.get("text")]
                )

                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(
                    model="timHan/llama3korean8B4QKM:latest",
                    base_url="http://127.0.0.1:11434",
                    max_tokens=300
                )

                prompt = f"""
아래 문서 내용을 바탕으로 질문에 대해 정확하게 답변하세요.
문서에 없는 정보는 절대 추가하지 마세요.
중요 핵심만 2줄 이하로 요약
불필요한 설명, 부가 정보 금지
답변은 짧고 명확하게
요약 답변 앞에 어떤 설명도 쓰지마

질문: {q.question}
문서 내용: {context_text}

요약 답변:
"""
                response = llm.generate([prompt])
                answer = response.generations[0][0].text.strip()

                save_chat_history(
                    q.question,
                    answer,
                    f"{top_contexts[0]['file_name']} | {top_contexts[0].get('page_no','-')}",
                    session_id
                )

                return {
                    "session_id": session_id,
                    "answer": answer,
                    "source": f"{top_contexts[0]['file_name']} | {top_contexts[0].get('page_no','-')}"
                }

        # 예외 fallback (이론상 거의 안 탐)
        return {
            "session_id": session_id,
            "answer": "질문을 이해하지 못했습니다. 다시 한 번 구체적으로 질문해 주세요."
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

