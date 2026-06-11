"""
core.py — Polar-SSM + NeuralMarkov + XAI 통합 코어
====================================================
Layer 1: Polar-SSM 생성
Layer 2: NeuralMarkov 가드레일
Layer 3: XAI 설명
"""
from __future__ import annotations
import numpy as np
import pickle, time
from collections import Counter, defaultdict
from typing import Optional

from polar_engine import PolarEmbeddingEngine, tokenize
from polar_ssm import PolarSSMGenerator
from memory_engine import MemoryEngine


# ── NeuralMarkov ──────────────────────────────────────────────
class NMGuard:
    def __init__(self):
        self.uni=Counter(); self.bi=defaultdict(Counter)
        self.tri=defaultdict(Counter); self.total=0
        self.mu=0.0; self.std=1.0; self.is_trained=False

    def train(self, corpus, epochs=20):
        toks = tokenize(corpus)*epochs
        self.total = len(toks)
        for i,t in enumerate(toks):
            self.uni[t]+=1
            if i>=1: self.bi[toks[i-1]][t]+=1
            if i>=2: self.tri[(toks[i-2],toks[i-1])][t]+=1
        base = tokenize(corpus)
        scores=[self._s(base[i:i+5]) for i in range(0,len(base)-5,3)]
        if scores:
            self.mu=float(np.mean(scores))
            self.std=float(np.std(scores))+1e-12
        self.is_trained=True

    def _s(self, toks, alpha=0.001):
        V=len(self.uni)
        if V==0 or len(toks)<2: return -20.0
        lp=0.0; n=0
        for i in range(len(toks)):
            wc=toks[i]
            p1=(self.uni[wc]+alpha)/(self.total+alpha*V)
            p2=p3=0.0
            if i>=1:
                wp=toks[i-1]
                p2=self.bi[wp][wc]/self.uni[wp] if self.uni[wp]>0 else 0
            if i>=2:
                wpp=toks[i-2]
                p3=self.tri[(wpp,wp)][wc]/self.bi[wpp][wp] if self.bi[wpp][wp]>0 else 0
            lp+=np.log(0.6*p3+0.3*p2+0.1*p1+1e-12); n+=1
        return lp/max(n,1)

    def evaluate(self, text, thr=-11.5):
        toks = tokenize(text)
        s    = self._s(toks)
        st   = "PASS" if s>=-10 else "WARNING" if s>=thr else "FATAL"
        # 토큰별 logP
        V=len(self.uni); alpha=0.001
        token_scores=[]
        for i,wc in enumerate(toks):
            p1=(self.uni[wc]+alpha)/(self.total+alpha*V)
            p2=p3=0.0
            if i>=1:
                wp=toks[i-1]
                p2=self.bi[wp][wc]/self.uni[wp] if self.uni[wp]>0 else 0
            if i>=2:
                wpp=toks[i-2]
                p3=self.tri[(wpp,wp)][wc]/self.bi[wpp][wp] if self.bi[wpp][wp]>0 else 0
            tlp=np.log(0.6*p3+0.3*p2+0.1*p1+1e-12)
            token_scores.append((wc, float(tlp)))
        return {"status":st,"logp":s,"z":(s-self.mu)/self.std,
                "tokens":token_scores}


# ── 통합 코어 ─────────────────────────────────────────────────
class PolarAICore:
    """
    Polar-SSM + NM 가드레일 + XAI + 메모리
    모든 레이어 통합
    """
    def __init__(self):
        self.polar  = PolarEmbeddingEngine()
        self.ssm    = None   # PolarSSMGenerator
        self.guard  = NMGuard()
        self.memory = MemoryEngine()
        self.corpus = ""
        self.is_ready = False
        self.stats  = {"total":0,"ssm_pass":0,"fallback":0,"fatal":0}

    def build(self, corpus_text: str, epochs: int = 20,
              on_progress=None):
        self.corpus = corpus_text
        if on_progress: on_progress(10, "극성 임베딩 학습...")
        self.polar.train_corpus(corpus_text, window=4,
                                epochs=min(epochs,5))
        if on_progress: on_progress(40, "Polar-SSM 학습...")
        self.ssm = PolarSSMGenerator(self.polar)
        self.ssm.build(corpus_text, epochs=epochs)
        if on_progress: on_progress(75, "NM 가드레일 학습...")
        self.guard.train(corpus_text, epochs=epochs)
        if on_progress: on_progress(100, "완료")
        self.is_ready = True

    def answer(self, question: str,
               logp_thr: float = -11.5,
               use_memory: bool = True) -> dict:
        t0 = time.perf_counter()
        self.stats["total"] += 1

        if not self.is_ready:
            return self._empty("학습 필요", t0)

        # 맥락
        ctx = ""
        if use_memory and self.memory.current_id:
            ctx = self.memory.get_context_for_query(question)

        # Layer 1: Polar-SSM 생성
        r_ssm  = self.ssm.generate(question, max_tokens=15)
        ans_ssm= r_ssm["answer"]

        # Layer 2: NM 가드레일
        nm_r   = self.guard.evaluate(ans_ssm, thr=logp_thr)
        status = nm_r["status"]

        if status == "FATAL":
            # FATAL → 원본 문장 폴백
            answer = r_ssm["best_sent"]
            method = "fallback"
            self.stats["fallback"] += 1
        elif status == "WARNING":
            # WARNING → 원본 문장이 더 좋으면 교체
            best = r_ssm["best_sent"]
            nm_best = self.guard.evaluate(best, thr=logp_thr)
            if nm_best["logp"] > nm_r["logp"]:
                answer = best; method = "fallback"
                self.stats["fallback"] += 1
            else:
                answer = ans_ssm; method = "ssm_warning"
                self.stats["ssm_pass"] += 1
        else:
            answer = ans_ssm; method = "ssm_pass"
            self.stats["ssm_pass"] += 1

        if not answer:
            self.stats["fatal"] += 1
            answer = "관련 정보가 없어요."
            method = "none"

        # Layer 3: XAI
        xai = self._make_xai(question, answer, r_ssm, nm_r)

        ms = (time.perf_counter()-t0)*1000

        # 메모리 저장
        if use_memory:
            if not self.memory.current_id:
                self.memory.new_session()
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer,
                                 quality=status, source=method)

        return {
            "answer":   answer,
            "method":   method,
            "status":   status,
            "logp":     nm_r["logp"],
            "xai":      xai,
            "state":    r_ssm.get("final_state", np.zeros(self.polar.N)),
            "ms":       ms,
        }

    def _make_xai(self, question, answer, r_ssm, nm_r) -> dict:
        """XAI 설명 생성"""
        # 상태 설명
        state = r_ssm.get("final_state", np.zeros(self.polar.N))
        axes  = self.polar.axis_names[:self.polar.N]
        active= [(axes[i], float(state[i]))
                 for i in range(min(len(axes),len(state)))
                 if abs(state[i]) > 0.08]
        active.sort(key=lambda x:-abs(x[1]))
        state_desc = " | ".join(
            f"{a}{'↑' if v>0 else '↓'}{abs(v):.1f}"
            for a,v in active[:3]) or "중립"

        # 이탈 토큰
        outliers = [(t,lp) for t,lp in nm_r.get("tokens",[])
                    if lp < -12][:3]

        # 한줄 설명
        st = nm_r["status"]
        if st == "PASS":
            why = f"도메인 안 (logP:{nm_r['logp']:+.1f})"
        elif st == "WARNING":
            why = f"경계 수준 (logP:{nm_r['logp']:+.1f})"
        else:
            why = f"도메인 이탈 → 폴백 (logP:{nm_r['logp']:+.1f})"

        return {
            "state_desc": state_desc,
            "outliers":   outliers,
            "why":        why,
            "method":     r_ssm.get("method",""),
            "best_score": r_ssm.get("best_score",0),
        }

    def _empty(self, msg, t0):
        return {"answer":msg,"method":"none","status":"SKIP",
                "logp":-99,"xai":{},"state":np.zeros(self.polar.N),
                "ms":(time.perf_counter()-t0)*1000}

    # ── 저장/로드 ─────────────────────────────────────────────
    def to_bytes(self) -> bytes:
        data = {
            "polar_vecs":   {w:v.tolist() for w,v in self.polar.word_vecs.items()},
            "polar_conf":   dict(self.polar.confidence),
            "polar_src":    dict(self.polar.source),
            "polar_axes":   self.polar.axis_names,
            "polar_N":      self.polar.N,
            "polar_history":self.polar.history,
            "nm_uni":   dict(self.guard.uni),
            "nm_bi":    {k:dict(v) for k,v in self.guard.bi.items()},
            "nm_tri":   {k:dict(v) for k,v in self.guard.tri.items()},
            "nm_total": self.guard.total,
            "nm_mu":    self.guard.mu,
            "nm_std":   self.guard.std,
            "corpus":   self.corpus,
            "memory":   self.memory.to_dict(),
            "stats":    self.stats,
            # SSM 파라미터
            "ssm_C":    self.ssm.ssm.C.tolist() if self.ssm else None,
            "ssm_D":    self.ssm.ssm.D.tolist() if self.ssm else None,
            "ssm_A0":   self.ssm.ssm.A0 if self.ssm else 0.9,
            "ssm_B0":   self.ssm.ssm.B0 if self.ssm else 1.0,
            "ssm_vecs": {w:v.tolist()
                         for w,v in (self.ssm.vocab_vecs.items()
                                     if self.ssm else {})},
            "ssm_sents":[(s, v.tolist())
                          for s,v in (self.ssm.sent_index
                                      if self.ssm else [])],
        }
        return pickle.dumps(data)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "PolarAICore":
        data = pickle.loads(raw)
        core = cls()
        # 극성
        core.polar.word_vecs  = {w:np.array(v)
                                  for w,v in data["polar_vecs"].items()}
        core.polar.confidence = data["polar_conf"]
        core.polar.source     = data["polar_src"]
        core.polar.axis_names = data["polar_axes"]
        core.polar.N          = data["polar_N"]
        core.polar.history    = data.get("polar_history",[])
        # NM
        core.guard.uni   = Counter(data["nm_uni"])
        core.guard.bi    = defaultdict(Counter,
                            {k:Counter(v) for k,v in data["nm_bi"].items()})
        core.guard.tri   = defaultdict(Counter,
                            {k:Counter(v) for k,v in data["nm_tri"].items()})
        core.guard.total = data["nm_total"]
        core.guard.mu    = data["nm_mu"]
        core.guard.std   = data["nm_std"]
        core.guard.is_trained = True
        # 코퍼스 + 메모리
        core.corpus = data.get("corpus","")
        core.memory = MemoryEngine.from_dict(data.get("memory",{}))
        core.stats  = data.get("stats", core.stats)
        # SSM 복원
        if data.get("ssm_C") and core.corpus:
            core.ssm = PolarSSMGenerator(core.polar)
            core.ssm.build(core.corpus, epochs=1)  # 구조만
            core.ssm.ssm.C  = np.array(data["ssm_C"])
            core.ssm.ssm.D  = np.array(data["ssm_D"])
            core.ssm.ssm.A0 = data["ssm_A0"]
            core.ssm.ssm.B0 = data["ssm_B0"]
            core.ssm.vocab_vecs = {w:np.array(v)
                                    for w,v in data["ssm_vecs"].items()}
            core.ssm.sent_index = [(s,np.array(v))
                                    for s,v in data["ssm_sents"]]
        core.is_ready = True
        return core

    def summary(self) -> dict:
        return {
            "극성 어휘": len(self.polar.word_vecs),
            "NM 토큰":  self.guard.total,
            "문장 인덱스": len(self.ssm.sent_index) if self.ssm else 0,
            "대화 세션": len(self.memory.sessions),
            "전체 대화": self.memory.total_turns,
        }
