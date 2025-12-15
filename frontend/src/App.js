// --------------------------------------------------
// File: ~/RAG_Chatbot/frontend/src/App.js
// Description: 메인 패이지 컨포넌트, 채팅 버튼 및 ChatWindow 렌더링
// --------------------------------------------------

import React, { useState } from "react";
import "./App.css";
import ChatWindow from "./ChatWindow";

// ===== 상태 관리 =====
function App() {
  const [isOpen, setIsOpen] = useState(false);

// ===== 채팅창 토글 함수 =====
  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // ===== logo 불러오기 =====
  const logo = process.env.PUBLIC_URL + "/logo.png";

  return (
    <div className="App">
      {/* 오른쪽 하단 floating 버튼 */}
      <button className="chat-button" onClick={toggleChat}>
        <img src={logo} alt="chat logo" className="chat-logo" />
      </button>

      {/* 채팅창 */}
      {isOpen && <ChatWindow closeChat={toggleChat} />}
    </div>
  );
}

export default App;

