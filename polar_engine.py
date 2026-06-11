"""
polar_engine.py — 극성 스칼라 임베딩 엔진
==========================================
단어를 N차원 의미 축의 극성 스칼라로 표현
  매우부정 -1.5 / 부정 -1.0 / 중립 0.0 / 긍정 +1.0 / 매우긍정 +1.5

학습 단계:
  1. 시드: 반의어 쌍으로 축 정의
  2. 코퍼스: 공출현으로 미등록 단어 극성 전파
  3. 파인튜닝: 사용자 교정 + 새 쌍 추가

저장: pkl (word_vecs + axes + history + meta)
"""
from __future__ import annotations
import numpy as np
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

_rng = np.random.default_rng(42)

# ── 기본 의미 축 ──────────────────────────────────────────────
DEFAULT_AXES: dict[str, dict] = {
    "크기": {
        "pos": [("매우크다",1.5),("크다",1.0),("높다",1.0),("많다",1.0),
                ("넓다",0.8),("깊다",0.8),("길다",0.8)],
        "neg": [("매우작다",-1.5),("작다",-1.0),("낮다",-1.0),("적다",-1.0),
                ("좁다",-0.8),("얕다",-0.8),("짧다",-0.8)],
    },
    "속도": {
        "pos": [("매우빠르다",1.5),("빠르다",1.0),("신속하다",1.0),("즉각",0.8)],
        "neg": [("매우느리다",-1.5),("느리다",-1.0),("더디다",-0.8),("지연",-0.8)],
    },
    "감정": {
        "pos": [("매우좋다",1.5),("좋다",1.0),("기쁘다",1.0),("행복",1.0),
                ("만족",0.8),("긍정",0.8),("즐겁다",0.8)],
        "neg": [("매우나쁘다",-1.5),("나쁘다",-1.0),("슬프다",-1.0),("불만",-0.8),
                ("부정",-0.8),("우울",-0.8)],
    },
    "확실성": {
        "pos": [("확실하다",1.5),("명확하다",1.0),("분명하다",1.0),
                ("정확하다",1.0),("확신",0.8)],
        "neg": [("불확실하다",-1.5),("모호하다",-1.0),("애매하다",-1.0),
                ("불분명하다",-1.0),("의심",-0.8)],
    },
    "난이도": {
        "pos": [("매우쉽다",1.5),("쉽다",1.0),("간단하다",1.0),
                ("단순하다",0.8),("용이하다",0.8)],
        "neg": [("매우어렵다",-1.5),("어렵다",-1.0),("복잡하다",-1.0),
                ("난해하다",-0.8),("곤란하다",-0.8)],
    },
    "성취": {
        "pos": [("매우우수",1.5),("성공",1.0),("달성",1.0),("향상",0.9),
                ("발전",0.9),("우수",0.8),("합격",0.8)],
        "neg": [("매우불량",-1.5),("실패",-1.0),("저하",-0.9),("퇴보",-0.9),
                ("부족",-0.8),("불합격",-0.8)],
    },
    "양": {
        "pos": [("매우많다",1.5),("많다",1.0),("충분하다",0.8),
                ("풍부하다",0.8),("증가",0.8)],
        "neg": [("매우적다",-1.5),("적다",-1.0),("부족하다",-0.8),
                ("결핍",-0.8),("감소",-0.8)],
    },
    "시간": {
        "pos": [("빠르다",1.0),("이르다",0.8),("신속",0.8),("조기",0.7)],
        "neg": [("늦다",-1.0),("지연",-0.8),("지체",-0.8),("만료",-0.5)],
    },
    "중요성": {
        "pos": [("매우중요",1.5),("중요하다",1.0),("핵심",1.0),
                ("필수",0.9),("필요",0.7)],
        "neg": [("불필요",-1.0),("사소하다",-0.8),("부차적",-0.7),
                ("무관하다",-0.8)],
    },
    "효율": {
        "pos": [("매우효율",1.5),("효율적",1.0),("효과적",1.0),
                ("최적",0.9),("경량",0.7)],
        "neg": [("비효율",-1.0),("낭비",-0.8),("과부하",-0.7),
                ("무겁다",-0.6)],
    },
}


# ── 토크나이저 ─────────────────────────────────────────────────
_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","없습니다","입니다",
         "했다","한다","이다","하고","해서","하여","되어","이며",
         "하는","된","한","이고","인가","인지"]

def tokenize(text: str) -> list:
    tokens = []
    for word in text.replace("\n"," ").split():
        word = word.strip(".,!?()[]\"'~?：:；;")
        stem = word
        for s in sorted(_JOSA+_EOMI, key=len, reverse=True):
            if word.endswith(s) and len(word)>len(s)+1:
                stem = word[:-len(s)]; break
        if stem and len(stem) > 1:
            tokens.append(stem)
    return tokens


# ── 파인튜닝 히스토리 ─────────────────────────────────────────
@dataclass
class FinetuneRecord:
    type:       str   # "pair" / "correct" / "corpus"
    data:       dict
    timestamp:  float = field(default_factory=time.time)
    note:       str = ""


# ── 극성 임베딩 엔진 ─────────────────────────────────────────
class PolarEmbeddingEngine:
    """
    극성 스칼라 임베딩 엔진
    N차원 의미 축 × 극성 스칼라 (-1.5 ~ +1.5)
    """
    VERSION = "1.0"

    def __init__(self, axes: dict = None):
        self.axes       = axes or DEFAULT_AXES
        self.axis_names = list(self.axes.keys())
        self.N          = len(self.axis_names)

        # 핵심 데이터
        self.word_vecs:  dict[str, np.ndarray] = {}
        self.confidence: dict[str, float]      = {}  # 0~1 신뢰도
        self.source:     dict[str, str]        = {}  # seed/corpus/finetune

        # 학습 이력
        self.history: list[FinetuneRecord] = []
        self.corpus_tokens_seen = 0
        self.finetune_rounds    = 0

        # 메타
        self.created_at  = time.time()
        self.updated_at  = time.time()
        self.description = ""

        # 시드 초기화
        self._init_seeds()

    # ── 초기화 ───────────────────────────────────────────────
    def _init_seeds(self):
        """반의어 쌍으로 시드 벡터 설정"""
        for ax_idx, (ax_name, ax_data) in enumerate(self.axes.items()):
            for word, val in ax_data.get("pos", []):
                self._set_axis(word, ax_idx, val, source="seed")
            for word, val in ax_data.get("neg", []):
                self._set_axis(word, ax_idx, val, source="seed")

    def _set_axis(self, word: str, ax_idx: int, val: float,
                  source: str = "corpus", conf: float = 1.0,
                  merge_weight: float = 1.0):
        """특정 축의 극성값 설정 (기존값과 가중 병합)"""
        if word not in self.word_vecs:
            self.word_vecs[word]  = np.zeros(self.N)
            self.confidence[word] = 0.0
            self.source[word]     = source

        old = self.word_vecs[word][ax_idx]
        self.word_vecs[word][ax_idx] = (
            old * (1 - merge_weight) + val * merge_weight
        )
        self.word_vecs[word][ax_idx] = float(
            np.clip(self.word_vecs[word][ax_idx], -1.5, 1.5))

        # 신뢰도 갱신
        old_conf = self.confidence.get(word, 0.0)
        self.confidence[word] = min(1.0, old_conf + conf * 0.1)
        if source == "seed" or source == "finetune":
            self.confidence[word] = max(self.confidence[word], conf)
            self.source[word] = source

    # ── 코퍼스 학습 ──────────────────────────────────────────
    def train_corpus(self, corpus_text: str,
                     window: int = 4,
                     decay: float = 0.6,
                     epochs: int = 5,
                     on_progress=None) -> dict:
        """
        코퍼스에서 미등록 단어 극성 전파
        시드 단어와 공출현 → 극성 추정
        """
        tokens = tokenize(corpus_text) * epochs
        n = len(tokens)
        self.corpus_tokens_seen += len(tokenize(corpus_text))

        new_words  = 0
        updated    = 0
        iterations = 3  # 전파 반복

        for it in range(iterations):
            if on_progress:
                on_progress(int((it+1)/iterations*80), f"전파 {it+1}/{iterations}...")
            for i, w in enumerate(tokens):
                new_vals = defaultdict(list)
                for j in range(max(0,i-window), min(n,i+window+1)):
                    if j == i: continue
                    nbr = tokens[j]
                    if nbr not in self.word_vecs: continue
                    dist   = abs(i-j)
                    weight = decay ** dist
                    conf   = self.confidence.get(nbr, 0.0)
                    if conf < 0.1: continue
                    new_vals[w].append(
                        (self.word_vecs[nbr].copy(), weight * conf))

                if not new_vals[w]: continue
                total_w = sum(wt for _,wt in new_vals[w])
                if total_w < 1e-6: continue

                # 가중 평균 극성 벡터
                vec = sum(v*wt for v,wt in new_vals[w]) / total_w
                vec = np.clip(vec, -1.5, 1.5)

                is_new = w not in self.word_vecs
                # 시드는 코퍼스 학습에서 건너뜀 (보호)
                if not is_new and self.source.get(w) == "seed":
                    continue

                conf_gained = min(0.6, total_w / max(len(new_vals[w]),1))

                if is_new:
                    self.word_vecs[w]  = vec
                    self.confidence[w] = conf_gained
                    self.source[w]     = "corpus"
                    new_words += 1
                else:
                    # finetune 단어도 낮은 가중치로만 병합
                    mw = 0.2 if self.source.get(w) == "finetune" else 0.5
                    self.word_vecs[w] = np.clip(
                        self.word_vecs[w]*(1-mw) + vec*mw, -1.5, 1.5)
                    self.confidence[w] = min(1.0,
                        self.confidence.get(w,0)+conf_gained*0.1)
                    updated += 1

        self.updated_at = time.time()
        self.history.append(FinetuneRecord(
            type="corpus",
            data={"tokens":len(tokenize(corpus_text)),
                  "epochs":epochs, "new_words":new_words}
        ))

        if on_progress: on_progress(100, "완료")
        return {"new_words":new_words, "updated":updated,
                "total_vocab":len(self.word_vecs)}

    # ── 파인튜닝 ─────────────────────────────────────────────
    def add_antonym_pair(self, pos_word: str, neg_word: str,
                         axis_name: str, intensity: float = 1.0,
                         note: str = "") -> dict:
        """
        반의어 쌍 추가 / 교정
        기존 축에 추가하거나 새 축 생성
        """
        intensity = float(np.clip(intensity, 0.1, 1.5))

        # 축 찾기 또는 생성
        if axis_name not in self.axis_names:
            self.axis_names.append(axis_name)
            self.axes[axis_name] = {"pos":[], "neg":[]}
            # 기존 벡터에 새 차원 추가
            for w in self.word_vecs:
                self.word_vecs[w] = np.append(self.word_vecs[w], 0.0)
            self.N = len(self.axis_names)

        ax_idx = self.axis_names.index(axis_name)

        # 벡터 차원 맞추기
        for w in [pos_word, neg_word]:
            if w not in self.word_vecs:
                self.word_vecs[w]  = np.zeros(self.N)
                self.confidence[w] = 0.0
                self.source[w]     = "finetune"
            elif len(self.word_vecs[w]) < self.N:
                pad = self.N - len(self.word_vecs[w])
                self.word_vecs[w] = np.append(
                    self.word_vecs[w], np.zeros(pad))

        self._set_axis(pos_word, ax_idx, +intensity,
                       source="finetune", conf=1.0, merge_weight=0.8)
        self._set_axis(neg_word, ax_idx, -intensity,
                       source="finetune", conf=1.0, merge_weight=0.8)

        self.finetune_rounds += 1
        self.updated_at = time.time()
        self.history.append(FinetuneRecord(
            type="pair",
            data={"pos":pos_word,"neg":neg_word,
                  "axis":axis_name,"intensity":intensity},
            note=note,
        ))
        return {"status":"ok","axis":axis_name,"ax_idx":ax_idx}

    def correct_word(self, word: str, axis_name: str,
                     correct_val: float, note: str = "") -> dict:
        """단어 극성값 직접 교정"""
        if axis_name not in self.axis_names:
            return {"status":"axis_not_found"}
        ax_idx = self.axis_names.index(axis_name)
        correct_val = float(np.clip(correct_val, -1.5, 1.5))

        if word not in self.word_vecs:
            self.word_vecs[word]  = np.zeros(self.N)
            self.confidence[word] = 0.0
        elif len(self.word_vecs[word]) < self.N:
            self.word_vecs[word] = np.append(
                self.word_vecs[word], np.zeros(self.N-len(self.word_vecs[word])))

        self._set_axis(word, ax_idx, correct_val,
                       source="finetune", conf=1.0, merge_weight=0.9)
        self.updated_at = time.time()
        self.history.append(FinetuneRecord(
            type="correct",
            data={"word":word,"axis":axis_name,"val":correct_val},
            note=note,
        ))
        return {"status":"ok","word":word,"axis":axis_name,"val":correct_val}

    # ── 검색/유사도 ───────────────────────────────────────────
    def get_vec(self, word: str) -> Optional[np.ndarray]:
        v = self.word_vecs.get(word)
        if v is None: return None
        # 차원 맞추기
        if len(v) < self.N:
            v = np.append(v, np.zeros(self.N-len(v)))
        return v[:self.N]

    def similarity(self, w1: str, w2: str) -> float:
        """극성 코사인 유사도"""
        v1, v2 = self.get_vec(w1), self.get_vec(w2)
        if v1 is None or v2 is None: return 0.0
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12: return 0.0
        return float(np.dot(v1,v2)/(n1*n2))

    def nearest(self, word: str, topk: int = 10,
                source_filter: str = None) -> list:
        """가장 유사한 단어들"""
        v = self.get_vec(word)
        if v is None: return []
        sims = {}
        for w in self.word_vecs:
            if w == word: continue
            if source_filter and self.source.get(w) != source_filter: continue
            s = self.similarity(word, w)
            sims[w] = s
        return sorted(sims.items(), key=lambda x:-x[1])[:topk]

    def dominant_axes(self, word: str, topk: int = 5) -> list:
        """단어의 주요 의미 축"""
        v = self.get_vec(word)
        if v is None: return []
        axes = [(self.axis_names[i], float(v[i]))
                for i in range(min(self.N, len(v)))
                if abs(v[i]) > 0.05]
        return sorted(axes, key=lambda x:-abs(x[1]))[:topk]

    def polarity_spectrum(self, axis_name: str,
                          min_conf: float = 0.1) -> list:
        """특정 축에서 단어들의 극성 스펙트럼"""
        if axis_name not in self.axis_names: return []
        ax_idx = self.axis_names.index(axis_name)
        result = []
        for w, v in self.word_vecs.items():
            if self.confidence.get(w,0) < min_conf: continue
            if ax_idx < len(v) and abs(v[ax_idx]) > 0.05:
                result.append((w, float(v[ax_idx]),
                               self.confidence.get(w,0),
                               self.source.get(w,"")))
        return sorted(result, key=lambda x:x[1])

    # ── 저장/로드 ─────────────────────────────────────────────
    def save(self, path: str) -> int:
        """pkl로 저장 — 전체 상태 보존"""
        data = {
            "version":    self.VERSION,
            "axes":       self.axes,
            "axis_names": self.axis_names,
            "N":          self.N,
            "word_vecs":  {w:v.tolist() for w,v in self.word_vecs.items()},
            "confidence": dict(self.confidence),
            "source":     dict(self.source),
            "history":    self.history,
            "corpus_tokens_seen": self.corpus_tokens_seen,
            "finetune_rounds":    self.finetune_rounds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description":self.description,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return len(pickle.dumps(data))

    @classmethod
    def load(cls, path: str) -> "PolarEmbeddingEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        engine = cls.__new__(cls)
        engine.VERSION         = data.get("version","1.0")
        engine.axes            = data["axes"]
        engine.axis_names      = data["axis_names"]
        engine.N               = data["N"]
        engine.word_vecs       = {w:np.array(v)
                                  for w,v in data["word_vecs"].items()}
        engine.confidence      = data["confidence"]
        engine.source          = data["source"]
        engine.history         = data["history"]
        engine.corpus_tokens_seen = data.get("corpus_tokens_seen",0)
        engine.finetune_rounds = data.get("finetune_rounds",0)
        engine.created_at      = data.get("created_at",0)
        engine.updated_at      = data.get("updated_at",0)
        engine.description     = data.get("description","")
        return engine

    @classmethod
    def load_bytes(cls, raw: bytes) -> "PolarEmbeddingEngine":
        import io
        data = pickle.loads(raw)
        return cls.load.__func__(cls,
            type('F',(),{'read':lambda s:raw})())  # 바이트에서 직접

    # ── 통계 ─────────────────────────────────────────────────
    def summary(self) -> dict:
        by_source = Counter(self.source.values())
        return {
            "전체 어휘":     len(self.word_vecs),
            "의미 축":       self.N,
            "시드 단어":     by_source.get("seed",0),
            "코퍼스 학습":   by_source.get("corpus",0),
            "파인튜닝":      by_source.get("finetune",0),
            "파인튜닝 횟수": self.finetune_rounds,
            "코퍼스 토큰":   self.corpus_tokens_seen,
            "이력 수":       len(self.history),
        }
