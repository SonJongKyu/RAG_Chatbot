// --------------------------------------------------
// File: ~/RAG_Chatbot/frontend/src/api.js
// Description: FastAPI RAG 서버 API 호출용 함수 모음
// --------------------------------------------------

// ===== RAG 질의 =====
export async function queryRag(question, sessionId, forcedIntent = null) {
  try {
    const params = new URLSearchParams();

    if (sessionId) {
      params.append("session_id", sessionId);
    }

    if (forcedIntent) {
      params.append("forced_intent", forcedIntent);
    }

    const url =
      params.toString().length > 0
        ? `http://127.0.0.1:8601/rag_query?${params.toString()}`
        : `http://127.0.0.1:8601/rag_query`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    return await response.json();
  } catch (err) {
    console.error(err);
    return { error: "API 호출 실패" };
  }
}


// ===== 새 채팅 세션 생성 =====
export async function newChatSession() {
  const response = await fetch("http://127.0.0.1:8601/new_chat_session", { method: "POST" });
  return response.json();
}

// ===== 채팅 세션 삭제 =====
export async function deleteChatSession(sessionId) {
  const response = await fetch(`http://127.0.0.1:8601/delete_chat_session/${sessionId}`, { method: "DELETE" });
  return response.json();
}

// ===== 모든 채팅 세션 목록 가져오기 =====
export async function listChatSessions() {
  const response = await fetch("http://127.0.0.1:8601/list_chat_sessions");
  return response.json();
}

// ===== 특정 세션의 채팅 히스토리 가져오기 =====
export async function getChatHistory(sessionId) {
  const res = await fetch(`http://127.0.0.1:8601/get_chat_history/${sessionId}`);
  return res.json();
}

// ===== 버튼/시스템 메시지 저장 =====
export async function saveSystemMessage(message, sessionId, buttons = null) {
  try {
    const response = await fetch(`http://127.0.0.1:8601/save_system_message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, buttons })
    });
    return await response.json();
  } catch (err) {
    console.error(err);
    return { error: "API 호출 실패" };
  }
}
