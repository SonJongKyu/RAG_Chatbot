# --------------------------------------------------
# File: ~/RAG_Chatbot/Backend/intent_classifier.py
# Description: E5(SBERT 계열) 임베딩 기반 Intent 분류기
# --------------------------------------------------

import os
import json
from typing import Dict, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ===== Config =====
MODEL_NAME = "intfloat/multilingual-e5-small"
SIMILARITY_THRESHOLD = 0.62   # 이 값 미만이면 AMBIGUOUS
MARGIN_THRESHOLD = 0.06       # 1위-2위 점수 차이

# ===== Data structure =====
@dataclass
class IntentResult:
    intent: str
    confidence: float
    scores: Dict[str, float]

# ===== Utils =====
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count

# ===== Intent Classifier (SBERT / E5) =====
class E5IntentClassifier:
    def __init__(self, examples_by_intent: Dict[str, List[str]]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        self.examples_by_intent = examples_by_intent
        self.intent_embeddings = self._build_intent_embeddings()

    @torch.no_grad()
    def _embed(self, texts: List[str], prefix: str) -> torch.Tensor:

        texts = [f"{prefix}{t.strip()}" for t in texts]

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**tokens)
        embeddings = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )

        return F.normalize(embeddings, p=2, dim=1)
		# intent별 prototype embedding 생성
    def _build_intent_embeddings(self) -> Dict[str, torch.Tensor]:

        intent_vectors = {}

        for intent, examples in self.examples_by_intent.items():
            if not examples:
                continue

            embs = self._embed(examples, prefix="passage: ")
            proto = embs.mean(dim=0, keepdim=True)
            proto = F.normalize(proto, p=2, dim=1)
            intent_vectors[intent] = proto

        return intent_vectors

    @torch.no_grad()
    def classify(self, question: str) -> IntentResult:
        if not question or not question.strip():
            return IntentResult("AMBIGUOUS", 0.0, {})

        query_emb = self._embed([question], prefix="query: ")

        scores: Dict[str, float] = {}
        for intent, proto in self.intent_embeddings.items():
            score = (query_emb @ proto.T).item()
            scores[intent] = float(score)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        # 판단 기준
        if best_score < SIMILARITY_THRESHOLD:
            return IntentResult("AMBIGUOUS", best_score, scores)

        if (best_score - second_score) < MARGIN_THRESHOLD:
            return IntentResult("AMBIGUOUS", best_score, scores)

        return IntentResult(best_intent, best_score, scores)

# ===== Loader =====
_classifier_instance: E5IntentClassifier | None = None

def load_intent_examples() -> Dict[str, List[str]]:
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "intent_examples.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "intent_examples.json 파일이 없습니다. "
            "Backend/intent_examples.json 을 생성하세요."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_classifier() -> E5IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        examples = load_intent_examples()
        _classifier_instance = E5IntentClassifier(examples)
    return _classifier_instance

# ===== Public API (main.py에서 호출) =====
def classify_intent(question: str) -> dict:

    clf = get_classifier()
    result = clf.classify(question)

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "scores": result.scores
    }
