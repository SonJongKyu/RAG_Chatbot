# 🧠 RAG_Chatbot

한국어 LLM 기반 **RAG 챗봇 프로젝트**입니다.
PDF 문서를 임베딩하여 검색 후, 질문에 대한 답변을 제공합니다.
WSL, Python, FastAPI, Ollama 기반으로 동작하며, PDF 업로드 → FAISS 임베딩 → RAG 질의응답까지 지원합니다.

---

## 버전/스펙

**운영체제(OS)**
- WSL/Windows/Linux 등
- OS 버전
  
```bash
운영체제: Windows 11 + WSL2, Ubuntu 24.04 LTS
```

**Pyhton버전**
- Python 버전 명시
- 패키지 설치 및 가상환경 정보

```bash
Python: 3.12.1  
가상환경: venv
```

**주요 라이브러리**
- FastAPI, Uvicorn, Ollama, SentenceTransformers 등
- pip freeze로 버전 기록 가능

```bash
fastapi==0.109.1
uvicorn==0.23.2
sentence-transformers==2.2.2
ollama==0.13.0
```

**하드웨어/스펙 정보**
- CPU, RAM 등

```bash
하드웨어 권장 사양:
- CPU: Intel i5 이상
- RAM: 16GB 이상
- SSD 권장
- Ollama 모델 실행 시 인터넷 연결 필요
```

**기타 환경 변수**
- 프로젝트 경로, BASE_DIR, FAISS 저장 경로 등

```bash
프로젝트 경로: ~/RAG_Chatbot
Backend 경로:  ~/RAG_Chatbot/Backend/
Frontend 경로:  ~/RAG_Chatbot/frontend/src/
FAISS DB 위치: ~/RAG_Chatbot/faiss_db/
```

---

## 🖥 환경 준비 (WSL + Ubuntu)

Windows 환경에서 WSL 설치 및 Ubuntu 24.04 설정:

```powershell
# 관리자 권한 PowerShell
wsl --install -d Ubuntu-24.04 --name AI_KnowledgeOps
wsl -d AI_KnowledgeOps
```

Ubuntu에서 필수 패키지 설치:

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip curl wget git vim tree net-tools
```

---

## 🛠 VS Code 설정 (WSL 확장)

1. VS Code 열기 → **Extensions(확장)** 클릭
2. **WSL** 검색 후 설치
3. **명령 팔레트(Ctrl+Shift+P)** → `WSL: Connect to WSL` 선택

---

## 📂 프로젝트 클론

```bash
# 홈 디렉토리 이동
cd ~

# GitHub에서 프로젝트 클론
git clone https://github.com/SonJongKyu/RAG_Chatbot.git
cd RAG_Chatbot
```

---

## 🐍 Python 가상환경 및 패키지 설치

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

---

## 🤖 Ollama 설치 및 모델 다운로드

- 인터넷이 가능할 경우

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 설치 확인
ollama --version

# 설치 가능한 모델 확인
ollama list

# 한국어 LLaMA 모델 다운로드
ollama pull timHan/llama3korean8B4QKM:latest

# 다운로드 완료 확인
ollama list
```

---

## 🚀 백엔드 서버 실행

```bash
cd Backend
uvicorn main:app --host 0.0.0.0 --port 8601 --reload
```

* 서버 정상 실행 시:
  [http://localhost:8601](http://localhost:8601) → `{"status":"ok"}` 확인

---

## 🌐 프론트엔드 개발 서버 실행

```bash
# Node.js/NPM 설치
sudo apt install npm

# 패키지 설치
npm install

# 개발 서버 실행
npm start
```

* 브라우저에서 [http://localhost:3000](http://localhost:3000) 접속 → UI 확인

---

## 📁 프로젝트 구조

```text
RAG_Chatbot/
├─ Backend/                # FastAPI 서버 코드
├─ chat_history_sessions/   # 세션 기록 (JSON)
├─ faiss_db/               # FAISS 벡터 DB
├─ input/                  # 업로드 PDF
├─ output/                 # 로그 등
├─ frontend/               # React 프론트엔드
└─ venv/                   # Python 가상환경
```

---

## 🧪 실제 테스트

1) 터미널

```bash
curl -X POST "http://localhost:8601/rag_query" \
-H "Content-Type: application/json" \
-d '{"question":"금융기관이 뭐야?"}'
```

2) 웹 페이지

- 새로운 세션 생성 후
- 질문 입력: 금융기관이 뭐야?
