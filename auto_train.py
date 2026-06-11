"""
auto_train.py — SOM 의미 지도 + 자동 학습
==========================================
SOM 시각화 → 클러스터 클릭 → 관련 코퍼스 자동 생성
→ 10분 반복 학습 → pkl 저장
"""
from __future__ import annotations
import numpy as np
import time
import json
from collections import defaultdict
from typing import Optional

_rng = np.random.default_rng(42)


# ── SOM 시각화용 빌더 ─────────────────────────────────────────
class SOMBuilder:
    """
    극성 임베딩 → SOM 2D 지도 빌드
    x축: 극성 (-1.5 ~ +1.5)
    y축: 의미 도메인 (분리도 기반)
    """
    def __init__(self, grid: int = 8):
        self.grid = grid
        self.W    = None
        self.neuron_data: dict[int, list] = {}

    def build_from_engine(self, engine):
        """극성 임베딩 엔진에서 SOM 빌드"""
        words = [w for w,v in engine.word_vecs.items()
                 if engine.confidence.get(w,0) >= 0.1]
        if len(words) < 4:
            return False

        N = engine.N
        vecs = np.array([engine.word_vecs[w][:N] for w in words])
        # 차원 정규화
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = np.where(norms>1e-12, vecs/norms, vecs)

        # SOM 초기화 (극성 1차 축 기준)
        self.W = np.zeros((self.grid*self.grid, N))
        for i in range(self.grid*self.grid):
            gi, gj = i//self.grid, i%self.grid
            self.W[i,0] = (gj/(self.grid-1))*3.0-1.5
            if N>1: self.W[i,1] = (gi/(self.grid-1))*2.0-1.0
            self.W[i] += _rng.normal(0, 0.05, N)

        # SOM 학습
        n = len(vecs)
        for ep in range(50):
            lr    = 0.5*np.exp(-ep/50)
            sigma = max(0.5, self.grid/2*np.exp(-ep/50))
            for i in _rng.permutation(n):
                diff = self.W - vecs[i]
                bmu  = int(np.argmin(np.linalg.norm(diff, axis=1)))
                gi,gj = bmu//self.grid, bmu%self.grid
                for j in range(self.grid*self.grid):
                    ngi,ngj = j//self.grid, j%self.grid
                    h = np.exp(-((gi-ngi)**2+(gj-ngj)**2)/(2*sigma**2))
                    self.W[j] += lr*h*(vecs[i]-self.W[j])

        # 배치
        self.neuron_data = {}
        for w, v in zip(words, vecs):
            diff = self.W - v
            bmu  = int(np.argmin(np.linalg.norm(diff, axis=1)))
            if bmu not in self.neuron_data:
                self.neuron_data[bmu] = []
            self.neuron_data[bmu].append({
                "word":       w,
                "polarity":   float(engine.word_vecs[w][0]) if N>0 else 0.0,
                "confidence": float(engine.confidence.get(w,0)),
                "source":     engine.source.get(w,"?"),
                "axes":       engine.dominant_axes(w, topk=2),
            })
        return True

    def get_map_json(self, engine=None) -> list:
        """Streamlit HTML 렌더링용 JSON"""
        result = []
        for idx, items in self.neuron_data.items():
            gi, gj = idx//self.grid, idx%self.grid
            avg_pol = np.mean([it["polarity"] for it in items])
            result.append({
                "idx":    idx,
                "gi":     gi,
                "gj":     gj,
                "words":  [it["word"] for it in items[:5]],
                "count":  len(items),
                "avg_pol": float(avg_pol),
                "sources": list(set(it["source"] for it in items)),
            })
        return result

    def get_cluster_words(self, neuron_idx: int,
                          engine, topk: int = 20) -> list:
        """특정 뉴런 + 인접 뉴런의 단어들"""
        gi, gj = neuron_idx//self.grid, neuron_idx%self.grid
        words = []
        for idx, items in self.neuron_data.items():
            ngi, ngj = idx//self.grid, idx%self.grid
            dist = abs(gi-ngi)+abs(gj-ngj)
            if dist <= 1:
                words.extend(it["word"] for it in items)
        # 유사어도 포함
        for w in words[:5]:
            nears = engine.nearest(w, topk=5)
            words.extend(ww for ww,_ in nears if ww not in words)
        return list(dict.fromkeys(words))[:topk]


# ── 코퍼스 자동 생성기 ────────────────────────────────────────
class AutoCorpusGenerator:
    """
    클러스터 단어들로 학습용 코퍼스 자동 생성
    LLM 있으면 LLM, 없으면 패턴 기반
    """
    # 패턴 기반 문장 템플릿
    PATTERNS = [
        "{a}은 {b}와 관련이 있다",
        "{a}와 {b}는 비슷한 특성을 가진다",
        "{a}이 높으면 {b}도 높다",
        "{a}의 수준은 {b}의 정도를 나타낸다",
        "{a}과 {b}를 함께 고려해야 한다",
        "{a}은 {b}를 통해 평가할 수 있다",
        "{a}이 {b}에 영향을 미친다",
        "{a}의 향상은 {b}의 발전을 이끈다",
        "{b}의 변화는 {a}과 연관된다",
        "{a}을 개선하면 {b}도 향상된다",
    ]

    def generate_pattern(self, words: list,
                         n_sentences: int = 50) -> str:
        """패턴 기반 코퍼스 생성"""
        if len(words) < 2:
            return " ".join(words) * 20
        sentences = []
        words = [w for w in words if len(w) > 1]
        for _ in range(n_sentences):
            pat = _rng.choice(self.PATTERNS)
            idx_a = int(_rng.integers(0, len(words)))
            idx_b = int(_rng.integers(0, len(words)))
            while idx_b == idx_a and len(words) > 1:
                idx_b = int(_rng.integers(0, len(words)))
            sentences.append(pat.format(a=words[idx_a], b=words[idx_b]))
        # 단어 반복 (공출현 강화)
        base = " ".join(words) + " "
        return base*10 + "\n" + "\n".join(sentences)

    def generate_llm(self, words: list, api_key: str,
                     model: str = "gpt-4o-mini") -> str:
        """LLM 기반 코퍼스 생성"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"다음 개념들과 관련된 한국어 문장을 30개 만들어줘. "
                f"각 문장은 1~2개 개념을 포함해야 해. "
                f"개념들: {', '.join(words[:15])}\n\n"
                f"형식: 한 줄에 문장 하나. 설명 없이 문장만."
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=800,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self.generate_pattern(words)


# ── 자동 학습 루프 ────────────────────────────────────────────
class AutoTrainer:
    """
    10분 자동 학습 루프
    클러스터 선택 → 코퍼스 생성 → 학습 → 반복
    """
    def __init__(self, engine, som_builder: SOMBuilder,
                 duration_sec: int = 600):
        self.engine   = engine
        self.som      = som_builder
        self.duration = duration_sec
        self.gen      = AutoCorpusGenerator()

        # 통계
        self.rounds_done   = 0
        self.words_added   = 0
        self.start_time    = 0.0
        self.is_running    = False
        self.log: list[str] = []

    def train_round(self, neuron_idx: int, api_key: str = "",
                    model: str = "gpt-4o-mini",
                    epochs_per_round: int = 3) -> dict:
        """한 라운드 학습"""
        cluster_words = self.som.get_cluster_words(
            neuron_idx, self.engine)

        # 코퍼스 생성
        if api_key:
            corpus = self.gen.generate_llm(cluster_words, api_key, model)
        else:
            corpus = self.gen.generate_pattern(cluster_words, n_sentences=40)

        # 학습
        result = self.engine.train_corpus(
            corpus, window=3, epochs=epochs_per_round)

        self.rounds_done += 1
        self.words_added += result["new_words"]

        msg = (f"라운드 {self.rounds_done}: "
               f"+{result['new_words']}어휘 | "
               f"전체 {result['total_vocab']}개")
        self.log.append(msg)
        return {**result, "cluster_words": cluster_words[:8], "corpus_len": len(corpus)}

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed())

    def progress(self) -> float:
        return min(1.0, self.elapsed() / self.duration)
