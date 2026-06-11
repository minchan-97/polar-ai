"""
polar_ssm.py — Polar State Space Model
========================================
Mamba SSM 수식 + 극성 스칼라 임베딩

h(t) = A(x)·h(t-1) + B(x)·x(t)   ← 선택적 상태 업데이트
y(t) = C·h(t) + D·x(t)            ← 출력

핵심 차별점:
  x(t): 극성 벡터 (의미 있는 차원)
  A(x): 극성 강도에 따라 망각/기억 결정
  B(x): 극성 강도에 따라 입력 게이팅
  → 상태 h(t)가 해석 가능한 극성 벡터
"""
from __future__ import annotations
import numpy as np
import time
from collections import Counter, defaultdict
from typing import Optional

_rng = np.random.default_rng(42)

_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했다","한다","이다","하고","해서","하여","되어","이며",
         "하는","된","한","이고","했습니다","합니다","됩니다"]

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


# ── Polar-SSM 코어 ────────────────────────────────────────────
class PolarSSM:
    """
    극성 상태 공간 모델
    
    A(x): 선택적 망각 게이트
      |극성| 클수록 → 기억 강화 (A → 1)
      |극성| 작을수록 → 망각 (A → 0.5)
    
    B(x): 선택적 입력 게이트
      |극성| 클수록 → 강하게 상태에 반영
      중립 단어 → 약하게 반영
    """
    def __init__(self, N: int):
        self.N   = N                       # 극성 축 수
        # 파라미터 초기화
        self.C   = np.eye(N)               # 출력 투영
        self.D   = np.zeros(N)             # 스킵 연결
        self.A0  = 0.9                     # 기본 망각 강도
        self.B0  = 1.0                     # 기본 입력 강도
        self.is_trained = False

    def selective_A(self, x: np.ndarray) -> np.ndarray:
        """
        입력 의존 망각 게이트 (대각 행렬)
        |x_i| 클수록 A_ii → A0 (기억)
        |x_i| 작을수록 A_ii → 0.5 (망각)
        """
        strength = np.abs(x) / 1.5        # 0~1 정규화
        a_diag   = 0.5 + (self.A0-0.5) * strength
        return a_diag                       # 대각 원소만

    def selective_B(self, x: np.ndarray) -> np.ndarray:
        """
        입력 의존 입력 게이트
        |x_i| 클수록 강하게 반영
        """
        strength = np.abs(x) / 1.5
        return self.B0 * (0.2 + 0.8*strength)

    def step(self, h: np.ndarray,
             x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """단일 SSM 스텝"""
        a = self.selective_A(x)    # [N]
        b = self.selective_B(x)    # [N]
        # h(t) = A⊙h(t-1) + B⊙x(t)  (원소별 곱 = 경량 대각 SSM)
        h_new = a * h + b * x
        h_new = np.clip(h_new, -1.5, 1.5)
        # y(t) = C·h(t) + D⊙x(t)
        y = self.C @ h_new + self.D * x
        return h_new, y

    def forward(self, vecs: list) -> tuple[list, list]:
        """시퀀스 처리 → (출력 시퀀스, 상태 시퀀스)"""
        h = np.zeros(self.N)
        outputs, states = [], []
        for x in vecs:
            h, y = self.step(h, x)
            outputs.append(y.copy())
            states.append(h.copy())
        return outputs, states

    def train(self, token_seqs: list, polar_engine,
              lr: float = 0.05, epochs: int = 20):
        """
        C, D 파라미터 학습
        다음 토큰 극성 예측 손실 최소화
        GPU 없이 numpy gradient descent
        """
        N = self.N

        for ep in range(epochs):
            grad_C = np.zeros((N,N))
            grad_D = np.zeros(N)
            total_loss = 0.0
            n_steps    = 0

            for tokens in token_seqs:
                vecs = [self._get_vec(t, polar_engine)
                        for t in tokens]
                if len(vecs) < 2: continue

                h = np.zeros(N)
                for i in range(len(vecs)-1):
                    x     = vecs[i]
                    x_nxt = vecs[i+1]        # 다음 토큰 (목표)
                    h, y  = self.step(h, x)

                    # 손실: MSE(y, x_next)
                    err   = y - x_nxt
                    loss  = float(np.sum(err**2))
                    total_loss += loss; n_steps += 1

                    # 그라디언트
                    grad_C += np.outer(err, h)
                    grad_D += err * x

            if n_steps == 0: continue
            # 파라미터 업데이트
            self.C -= lr * grad_C / n_steps
            self.D -= lr * grad_D / n_steps
            # C 정규화 (발산 방지)
            self.C = np.clip(self.C, -2.0, 2.0)

        self.is_trained = True
        return {"loss": total_loss/max(n_steps,1)}

    def _get_vec(self, token: str, polar_engine) -> np.ndarray:
        v = polar_engine.get_vec(token)
        if v is None: return np.zeros(self.N)
        return v[:self.N] if len(v)>=self.N else np.pad(v,(0,self.N-len(v)))


# ── Polar-SSM 생성기 ──────────────────────────────────────────
class PolarSSMGenerator:
    """
    Polar-SSM 기반 텍스트 생성
    
    질문 → 극성 상태 h(t) 추적
    → 출력 y(t)와 가장 가까운 토큰 선택
    → 의미 일관성 있는 답변 생성
    """
    def __init__(self, polar_engine, n_axes: int = None):
        self.polar = polar_engine
        self.N     = n_axes or polar_engine.N
        self.ssm   = PolarSSM(self.N)
        self.vocab_vecs: dict[str, np.ndarray] = {}
        self.sent_index: list[tuple] = []  # (문장, 벡터)
        self.markov_bi  = defaultdict(Counter)  # 폴백
        self.markov_uni = Counter()
        self.is_ready   = False

    def build(self, corpus_text: str, epochs: int = 20):
        """코퍼스로 Polar-SSM 학습"""
        sentences = [s.strip() for s in corpus_text.split("\n")
                     if s.strip() and len(s.strip())>4]
        all_tokens = tokenize(corpus_text)

        # 어휘 빈도
        tok_freq = Counter(all_tokens)

        # 어휘 벡터 구축 — 극성 없는 단어는 TF 기반 더미 벡터
        for t in set(all_tokens):
            v = self.polar.get_vec(t)
            if v is not None and np.linalg.norm(v) > 1e-12:
                self.vocab_vecs[t] = v[:self.N] if len(v)>=self.N \
                                     else np.pad(v,(0,self.N-len(v)))
            else:
                # 극성 없는 단어: 빈도 기반 더미 (첫 번째 축에 작은 값)
                dummy = np.zeros(self.N)
                dummy[0] = min(tok_freq[t] / 50.0, 0.3)  # 크기 축에 소량
                self.vocab_vecs[t] = dummy

        # TF-IDF 어휘 / IDF 계산
        from collections import Counter as _C
        vocab = {w:i for i,w in enumerate(tok_freq.keys())}
        V     = len(vocab)
        N_s   = len(sentences)
        df    = _C()
        for s in sentences:
            for t in set(tokenize(s)):
                df[t] += 1
        idf = {w: np.log((N_s+1)/(df.get(w,0)+1))+1 for w in vocab}

        def tfidf_vec(text):
            v = np.zeros(V)
            toks = tokenize(text); cnt = _C(toks)
            for w,c in cnt.items():
                if w in vocab:
                    v[vocab[w]] = (c/max(len(toks),1))*idf.get(w,1.0)
            norm = np.linalg.norm(v)
            return v/norm if norm>1e-12 else v

        self._tfidf_vocab = vocab
        self._tfidf_vecs  = np.array([tfidf_vec(s) for s in sentences])
        self._tfidf_idf   = idf
        self._V           = V

        # 문장 인덱스 (극성 + TF-IDF 결합)
        for s in sentences:
            toks = tokenize(s)
            if not toks: continue
            # 극성 벡터 평균
            pvecs = [self.vocab_vecs[t] for t in toks
                     if t in self.vocab_vecs]
            if pvecs:
                pv = np.mean(pvecs, axis=0)
                norm = np.linalg.norm(pv)
                pv = pv/norm if norm>1e-12 else pv
            else:
                pv = np.zeros(self.N)
            self.sent_index.append((s, pv))

        # 마르코프 폴백
        for i, t in enumerate(all_tokens):
            self.markov_uni[t] += 1
            if i>=1: self.markov_bi[all_tokens[i-1]][t] += 1

        # SSM 학습
        token_seqs = [tokenize(s) for s in sentences if len(tokenize(s))>=3]
        if token_seqs:
            self.ssm.train(token_seqs, self.polar,
                          lr=0.01, epochs=epochs)
        self.is_ready = True

    def generate(self, question: str,
                 max_tokens: int = 20) -> dict:
        t0 = time.perf_counter()
        if not self.is_ready:
            return {"answer":"","method":"none","ms":0,
                    "final_state":np.zeros(self.N),"best_score":0}

        q_toks = tokenize(question)

        # 1. SSM으로 문맥 상태
        q_vecs = [self.vocab_vecs.get(t, np.zeros(self.N))
                  for t in q_toks]
        _, states = self.ssm.forward(q_vecs)
        final_state = states[-1] if states else np.zeros(self.N)

        # 2. 하이브리드 검색: TF-IDF(주) + 극성 상태(보조)
        best_sent  = ""
        best_score = -1.0

        # TF-IDF 유사도
        def _tv(tokens):
            from collections import Counter as _C
            v = np.zeros(self._V)
            cnt = _C(tokens)
            for w,c in cnt.items():
                if w in self._tfidf_vocab:
                    v[self._tfidf_vocab[w]] = (c/max(len(tokens),1))*self._tfidf_idf.get(w,1.0)
            norm = np.linalg.norm(v)
            return v/norm if norm>1e-12 else v

        qv_tfidf = _tv(q_toks)
        tfidf_sims = self._tfidf_vecs @ qv_tfidf

        # 극성 상태 유사도
        s_norm = np.linalg.norm(final_state)
        if s_norm > 1e-12:
            polar_sims = np.array([
                float(np.dot(final_state/s_norm, sv))
                for _, sv in self.sent_index
            ])
        else:
            polar_sims = np.zeros(len(self.sent_index))

        # 결합 (TF-IDF 70% + 극성 30%)
        combined = 0.7*tfidf_sims + 0.3*polar_sims
        best_idx  = int(np.argmax(combined))
        best_score= float(combined[best_idx])
        best_sent = self.sent_index[best_idx][0] if self.sent_index else ""

        # 3. SSM으로 생성
        if best_sent and best_score > 0.05:
            seed_toks = tokenize(best_sent)[:3]
            generated = self._ssm_generate(seed_toks, final_state, max_tokens)
            method    = "polar_ssm"
        else:
            generated = best_sent
            method    = "retrieval"

        ms = (time.perf_counter()-t0)*1000
        return {
            "answer":      generated or best_sent,
            "best_sent":   best_sent,
            "method":      method,
            "state_norm":  float(s_norm),
            "best_score":  best_score,
            "final_state": final_state,
            "ms":          ms,
        }

    def _ssm_generate(self, seed: list,
                      context_h: np.ndarray,
                      max_len: int) -> str:
        """SSM 상태에서 토큰 생성 — 반복 방지"""
        result = list(seed)
        h      = context_h.copy()
        seen_bigrams = set()

        for _ in range(max_len - len(seed)):
            curr_tok = result[-1]
            x = self.vocab_vecs.get(curr_tok, np.zeros(self.N))
            h, y = self.ssm.step(h, x)

            # 최근 5개 토큰 제외 + 이미 본 바이그램 제외
            recent   = set(result[-5:])
            nxt = self._nearest_token(y, exclude=recent)
            if not nxt: break

            # 바이그램 반복 방지
            bigram = (curr_tok, nxt)
            if bigram in seen_bigrams: break
            seen_bigrams.add(bigram)

            result.append(nxt)

            # 자연스러운 종결
            if any(result[-1].endswith(e)
                   for e in ["다","요","임","됨","있다","없다","이다"]):
                if len(result) >= 4: break

        # 최대 10토큰으로 제한 (깔끔하게)
        return " ".join(result[:10])

    def _nearest_token(self, target: np.ndarray,
                       exclude: set = None) -> str:
        """극성 공간에서 가장 가까운 토큰"""
        exclude = exclude or set()
        best_tok = ""; best_sim = -2.0
        t_norm   = np.linalg.norm(target)
        if t_norm < 1e-12: return ""

        for tok, v in self.vocab_vecs.items():
            if tok in exclude: continue
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12: continue
            sim = float(np.dot(target, v)/(t_norm*v_norm))
            if sim > best_sim:
                best_sim = sim; best_tok = tok
        return best_tok

    def explain_state(self, state: np.ndarray) -> str:
        """상태 벡터 → 한국어 설명 (XAI)"""
        axes    = self.polar.axis_names[:self.N]
        active  = [(axes[i], float(state[i]))
                   for i in range(min(len(axes),len(state)))
                   if abs(state[i]) > 0.1]
        active.sort(key=lambda x:-abs(x[1]))
        if not active: return "중립 상태"
        parts = []
        for ax, v in active[:3]:
            direction = "긍정" if v>0 else "부정"
            parts.append(f"{ax} {direction}({v:+.2f})")
        return " | ".join(parts)
