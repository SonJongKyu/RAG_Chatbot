// --------------------------------------------------
// File: ~/RAG_Chatbot/frontend/src/ChatSidebar.js
// Description: 사이드바 기능 + clarify 버튼 확장
// --------------------------------------------------

import React from "react";
import "./ChatWindow.css"; 

export default function ChatSidebar({
  sessions,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  currentSessionId,

  // ===== clarify 관련 props =====
  clarifyOptions = [],
  onSelectClarify
}) {
  return (
    <div className="session-list">
      {/* ===== 새 채팅 버튼 ===== */}
      <div className="new-chat-btn" onClick={onNewChat}>
        새 채팅
      </div>

      {/* ===== 세션 목록 스크롤 영역 ===== */}
      <div style={{ overflowY: "auto", flex: 1 }}>
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`session-item ${
              s.session_id === currentSessionId ? "active" : ""
            }`}
          >
            {/* ===== 세션 선택 영역 ===== */}
            <div
              style={{ flex: 1, cursor: "pointer" }}
              onClick={() => onSelectSession(s.session_id)}
            >
              {s.name || s.session_id}
            </div>

            {/* ===== 세션 삭제 버튼 ===== */}
            <button
              className="delete-btn"
              onClick={() => onDeleteSession(s.session_id)}
            >
              X
            </button>
          </div>
        ))}
      </div>

      {/* ===== clarify 버튼 영역 ===== */}
      {clarifyOptions && clarifyOptions.length > 0 && (
        <div className="clarify-box">
          <div className="clarify-title">
            어떤 정보를 원하시나요?
          </div>

          <div className="clarify-buttons">
            {clarifyOptions.map((opt, idx) => (
              <button
                key={idx}
                className="clarify-btn"
                onClick={() => {
                  if (onSelectClarify) {
                    onSelectClarify(opt.intent);
                  }
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
