"""
memory_engine.py — 대화 기억 + 맥락 관리
==========================================
대화 이력 + 맥락 요약 → pkl 저장/로드
사용자와의 대화를 기억하고 다음 답변에 활용
"""
from __future__ import annotations
import time
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했다","한다","이다","하고","해서","하여","되어","이며",
         "하는","된","한","이고"]

def tokenize(text):
    tokens = []
    for word in text.replace("\n"," ").split():
        word = word.strip(".,!?()[]\"'~?：:；;")
        stem = word
        for s in sorted(_JOSA+_EOMI, key=len, reverse=True):
            if word.endswith(s) and len(word)>len(s)+1:
                stem = word[:-len(s)]; break
        if stem and len(stem)>1: tokens.append(stem)
    return tokens


# ── 대화 턴 ──────────────────────────────────────────────────
@dataclass
class Turn:
    """단일 대화 턴"""
    role:      str           # "user" / "assistant"
    content:   str
    timestamp: float = field(default_factory=time.time)
    quality:   str = ""      # PASS/WARNING/FATAL
    source:    str = ""      # "offline" / "llm" / "none"
    keywords:  list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role":      self.role,
            "content":   self.content,
            "timestamp": self.timestamp,
            "quality":   self.quality,
            "source":    self.source,
            "keywords":  self.keywords,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(**d)


# ── 대화 세션 ────────────────────────────────────────────────
@dataclass
class Session:
    """하나의 대화 세션"""
    session_id:  str
    topic:       str = ""
    turns:       list = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)
    summary:     str = ""    # 세션 요약

    def add_turn(self, role: str, content: str, **kwargs) -> Turn:
        t = Turn(role=role, content=content, **kwargs)
        self.turns.append(t)
        return t

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "topic":      self.topic,
            "turns":      [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "summary":    self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(session_id=d["session_id"],
                topic=d.get("topic",""),
                created_at=d.get("created_at",0),
                summary=d.get("summary",""))
        s.turns = [Turn.from_dict(t) for t in d.get("turns",[])]
        return s


# ── 맥락 추적기 ───────────────────────────────────────────────
class ContextTracker:
    """
    대화 맥락 추적
    - 자주 언급된 키워드 → 관심 주제
    - 최근 질문 패턴 → 다음 검색 가중치
    - 미해결 질문 → 재시도 대상
    """
    def __init__(self, window: int = 10):
        self.window        = window  # 최근 N턴 기반
        self.keyword_freq  = Counter()
        self.topic_history = []
        self.unresolved    = []      # FATAL/없음 답변들

    def update(self, turn: Turn):
        """턴 추가 시 맥락 업데이트"""
        toks = tokenize(turn.content)
        for t in toks:
            self.keyword_freq[t] += 1
        if turn.role == "user":
            self.topic_history.append(toks)
        if (turn.role == "assistant" and
                turn.quality in ("FATAL","") and
                turn.source == "none"):
            self.unresolved.append(turn.content[:60])

    def get_context_keywords(self, topk: int = 10) -> list:
        """현재 맥락의 핵심 키워드"""
        return [w for w,_ in self.keyword_freq.most_common(topk)]

    def get_recent_topics(self, n: int = 3) -> list:
        """최근 대화 주제어"""
        recent = self.topic_history[-n:]
        flat   = [t for toks in recent for t in toks]
        cnt    = Counter(flat)
        return [w for w,_ in cnt.most_common(5)]

    def boost_query(self, query_tokens: list) -> list:
        """맥락 기반 쿼리 확장"""
        ctx_keywords = self.get_context_keywords(5)
        boosted = list(query_tokens)
        for kw in ctx_keywords:
            if kw not in boosted:
                boosted.append(kw)
        return boosted[:20]


# ── 메모리 엔진 ──────────────────────────────────────────────
class MemoryEngine:
    """
    대화 기억 + 맥락 관리 통합 엔진
    pkl로 전체 상태 저장/로드
    """
    def __init__(self, max_sessions: int = 50,
                 max_turns_per_session: int = 100):
        self.sessions:       dict[str, Session] = {}
        self.current_id:     Optional[str] = None
        self.context:        ContextTracker = ContextTracker()
        self.max_sessions    = max_sessions
        self.max_turns       = max_turns_per_session
        self.total_turns     = 0
        self.created_at      = time.time()
        self.user_name:      str = ""
        self.preferences:    dict = {}  # 사용자 선호 패턴

    # ── 세션 관리 ────────────────────────────────────────────
    def new_session(self, topic: str = "") -> str:
        sid = f"s{int(time.time()*1000)}"
        self.sessions[sid] = Session(session_id=sid, topic=topic)
        self.current_id    = sid
        # 오래된 세션 정리
        if len(self.sessions) > self.max_sessions:
            oldest = sorted(self.sessions.values(),
                            key=lambda s: s.created_at)[0]
            del self.sessions[oldest.session_id]
        return sid

    def get_current(self) -> Optional[Session]:
        if self.current_id:
            return self.sessions.get(self.current_id)
        return None

    def add_turn(self, role: str, content: str,
                 quality: str = "", source: str = "") -> Turn:
        if not self.current_id:
            self.new_session()
        sess  = self.sessions[self.current_id]
        toks  = tokenize(content)
        turn  = sess.add_turn(
            role, content,
            quality=quality,
            source=source,
            keywords=toks[:10],
        )
        self.context.update(turn)
        self.total_turns += 1
        # 최대 턴 수 초과 시 요약 후 정리
        if len(sess.turns) > self.max_turns:
            self._summarize_old_turns(sess)
        return turn

    def _summarize_old_turns(self, sess: Session):
        """오래된 턴을 요약으로 압축"""
        old_turns  = sess.turns[:20]
        sess.turns = sess.turns[20:]
        # 단순 키워드 요약
        all_text = " ".join(t.content for t in old_turns)
        toks     = tokenize(all_text)
        cnt      = Counter(toks)
        keywords = [w for w,_ in cnt.most_common(10)]
        summary  = f"[이전 대화 요약] 주요 주제: {', '.join(keywords)}"
        sess.summary = summary

    # ── 맥락 기반 검색 ───────────────────────────────────────
    def get_context_for_query(self, query: str,
                              window: int = 5) -> str:
        """
        현재 쿼리에 관련된 이전 대화 맥락 반환
        생성 엔진에 컨텍스트로 주입
        """
        sess = self.get_current()
        if not sess or not sess.turns:
            return ""

        q_toks = set(tokenize(query))
        relevant = []

        # 최근 N턴에서 관련 내용 추출
        recent = sess.turns[-window:]
        for turn in recent:
            t_toks = set(turn.keywords)
            overlap = q_toks & t_toks
            if overlap or turn.role == "user":
                relevant.append(f"{turn.role}: {turn.content[:80]}")

        if sess.summary:
            relevant.insert(0, sess.summary)
        return "\n".join(relevant[-5:])

    def get_related_history(self, query: str,
                            topk: int = 3) -> list:
        """
        모든 세션에서 유사 질문/답변 검색
        """
        q_toks = set(tokenize(query))
        results = []
        for sid, sess in self.sessions.items():
            for i, turn in enumerate(sess.turns):
                if turn.role != "user": continue
                t_toks  = set(turn.keywords)
                overlap = len(q_toks & t_toks)
                if overlap >= 2:
                    # 해당 질문의 답변 찾기
                    ans = ""
                    if i+1 < len(sess.turns):
                        ans = sess.turns[i+1].content[:80]
                    results.append({
                        "question": turn.content[:60],
                        "answer":   ans,
                        "overlap":  overlap,
                        "when":     turn.timestamp,
                    })
        results.sort(key=lambda x: (-x["overlap"], -x["when"]))
        return results[:topk]

    # ── 사용자 학습 ──────────────────────────────────────────
    def learn_preference(self, key: str, value: str):
        """사용자 선호 패턴 기억"""
        self.preferences[key] = value

    def get_preference(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    # ── 저장/로드 ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "sessions":    {sid: s.to_dict()
                            for sid, s in self.sessions.items()},
            "current_id":  self.current_id,
            "keyword_freq":dict(self.context.keyword_freq),
            "topic_history":self.context.topic_history[-50:],
            "unresolved":  self.context.unresolved[-20:],
            "total_turns": self.total_turns,
            "created_at":  self.created_at,
            "user_name":   self.user_name,
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEngine":
        mem = cls()
        mem.sessions = {
            sid: Session.from_dict(s)
            for sid, s in data.get("sessions",{}).items()
        }
        mem.current_id  = data.get("current_id")
        mem.total_turns = data.get("total_turns", 0)
        mem.created_at  = data.get("created_at", time.time())
        mem.user_name   = data.get("user_name", "")
        mem.preferences = data.get("preferences", {})
        # context 복원
        mem.context.keyword_freq  = Counter(data.get("keyword_freq",{}))
        mem.context.topic_history = data.get("topic_history",[])
        mem.context.unresolved    = data.get("unresolved",[])
        return mem

    def summary(self) -> dict:
        all_turns = sum(len(s.turns) for s in self.sessions.values())
        return {
            "세션 수":    len(self.sessions),
            "전체 턴":    self.total_turns,
            "현재 세션":  len(self.sessions.get(
                self.current_id or "", Session("")).turns),
            "관심 키워드": ", ".join(
                self.context.get_context_keywords(5)),
            "미해결 질문": len(self.context.unresolved),
        }
