"""
generation_engine.py — 오프라인 생성 엔진
==========================================
극성 임베딩 + 마르코프 → 오프라인 답변 생성
LLM 없이 완전 독립

생성 파이프라인:
  질문 분석 → 의도/핵심어 추출
      ↓
  극성 기반 관련 문장 검색
      ↓
  마르코프 전이로 답변 조합
      ↓
  NeuralMarkov 품질 검증
      ↓
  최종 답변
"""
from __future__ import annotations
import numpy as np
import re
import time
from collections import Counter, defaultdict
from typing import Optional

_rng = np.random.default_rng(42)

_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","없습니다","입니다",
         "했다","한다","이다","하고","해서","하여","되어","이며",
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


# ── 질문 분석기 ───────────────────────────────────────────────
class QueryAnalyzer:
    """
    질문 유형 분류 + 핵심어 추출
    """
    # 질문 유형 패턴
    TYPES = {
        "수치":   ["몇","얼마","몇일","몇시간","몇명","몇개","어느정도","how many","how much"],
        "시간":   ["언제","어느때","몇월","몇년","when"],
        "목록":   ["무엇","뭐","어떤","어느","목록","종류","what"],
        "여부":   ["있나요","있습니까","하나요","합니까","가능","is","are","do"],
        "이유":   ["왜","이유","원인","because","why"],
        "방법":   ["어떻게","방법","방식","절차","how"],
        "정의":   ["무엇인가","란","이란","정의","뜻","means","what is"],
        "비교":   ["차이","비교","versus","vs","compare"],
    }

    def analyze(self, question: str) -> dict:
        q = question.lower()
        toks = tokenize(question)

        # 질문 유형
        q_type = "일반"
        for type_name, patterns in self.TYPES.items():
            if any(p in q for p in patterns):
                q_type = type_name
                break

        # 핵심어 (질문어/조사 제거 후 남은 것)
        stopwords = set()
        for pats in self.TYPES.values():
            stopwords.update(pats)
        stopwords.update(["이것","그것","저것","이거","그거"])
        keywords = [t for t in toks
                    if t not in stopwords and len(t) > 1]

        return {
            "type":     q_type,
            "keywords": keywords,
            "tokens":   toks,
            "original": question,
        }


# ── 문장 검색기 ───────────────────────────────────────────────
class SentenceRetriever:
    """
    극성 임베딩 + TF-IDF 하이브리드 검색
    """
    def __init__(self):
        self.sentences: list[str] = []
        self.tfidf_vecs: Optional[np.ndarray] = None
        self.polar_vecs: Optional[np.ndarray] = None
        self.vocab: dict = {}
        self.idf:   dict = {}
        self.V = 0

    def build(self, corpus_text: str, engine=None):
        sents = list(dict.fromkeys(
            s.strip() for s in corpus_text.split("\n")
            if s.strip() and len(s.strip()) > 4
        ))
        self.sentences = sents

        # TF-IDF
        all_toks = tokenize(corpus_text)
        cnt = Counter(all_toks)
        self.vocab = {w:i for i,w in enumerate(
            w for w,c in cnt.most_common() if c >= 1)}
        self.V = len(self.vocab)
        N = len(sents)
        df = Counter()
        for s in sents:
            for t in set(tokenize(s)):
                if t in self.vocab: df[t] += 1
        self.idf = {w: np.log((N+1)/(df.get(w,0)+1))+1
                    for w in self.vocab}
        self.tfidf_vecs = np.array([self._tv(s) for s in sents])

        # 극성 벡터 (엔진 있으면)
        if engine and len(engine.word_vecs) > 0:
            self.polar_vecs = np.array([
                self._pv(s, engine) for s in sents])

    def _tv(self, text):
        v = np.zeros(self.V)
        toks = tokenize(text); cnt = Counter(toks)
        for w,c in cnt.items():
            if w in self.vocab:
                v[self.vocab[w]] = (c/max(len(toks),1))*self.idf.get(w,1.0)
        norm = np.linalg.norm(v)
        return v/norm if norm>1e-12 else v

    def _pv(self, text, engine):
        toks = tokenize(text)
        vecs = [engine.word_vecs[t]
                for t in toks if t in engine.word_vecs]
        if not vecs: return np.zeros(engine.N)
        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)
        return v/norm if norm>1e-12 else v

    def search(self, query_info: dict, engine=None,
               topk: int = 5, alpha: float = 0.7) -> list:
        """TF-IDF + 극성 결합 검색"""
        if self.V == 0: return []

        # TF-IDF
        expanded = query_info["keywords"] + query_info["tokens"]
        qv = self._tv_from_tokens(expanded)
        if np.linalg.norm(qv) < 1e-12:
            qv = self._tv(query_info["original"])

        tfidf_sim = self.tfidf_vecs @ qv if np.linalg.norm(qv)>1e-12 \
                    else np.zeros(len(self.sentences))

        # 극성 유사도
        if engine and self.polar_vecs is not None:
            q_pv = self._pv(query_info["original"], engine)
            n = np.linalg.norm(q_pv)
            if n > 1e-12:
                polar_sim = self.polar_vecs @ (q_pv/n)
            else:
                polar_sim = np.zeros(len(self.sentences))
            combined = alpha*tfidf_sim + (1-alpha)*polar_sim
        else:
            combined = tfidf_sim

        idx = np.argsort(-combined)[:topk]
        return [(self.sentences[i], float(combined[i]))
                for i in idx if combined[i] > 0.01]

    def _tv_from_tokens(self, tokens):
        v = np.zeros(self.V)
        cnt = Counter(tokens)
        for w,c in cnt.items():
            if w in self.vocab:
                v[self.vocab[w]] = (c/max(len(tokens),1))*self.idf.get(w,1.0)
        norm = np.linalg.norm(v)
        return v/norm if norm>1e-12 else v


# ── 마르코프 생성기 ───────────────────────────────────────────
class MarkovGenerator:
    """
    코퍼스 기반 문장 생성
    질문 유형별 생성 전략 적용
    """
    def __init__(self):
        self.uni   = Counter()
        self.bi    = defaultdict(Counter)
        self.tri   = defaultdict(Counter)
        self.total = 0

    def train(self, corpus: str, epochs: int = 20):
        toks = tokenize(corpus) * epochs
        self.total = len(toks)
        for i, t in enumerate(toks):
            self.uni[t] += 1
            if i>=1: self.bi[toks[i-1]][t] += 1
            if i>=2: self.tri[(toks[i-2],toks[i-1])][t] += 1

    def next_token(self, prev: str, prev2: str = None,
                   exclude: set = None) -> str:
        exclude = exclude or set()
        # 트라이그램 우선
        if prev2:
            tri_cands = {w:c for w,c in self.tri[(prev2,prev)].items()
                         if w not in exclude}
            if tri_cands:
                words = list(tri_cands.keys())
                probs = np.array(list(tri_cands.values()), dtype=float)
                probs /= probs.sum()
                return str(_rng.choice(words, p=probs))
        # 바이그램
        bi_cands = {w:c for w,c in self.bi[prev].items()
                    if w not in exclude}
        if bi_cands:
            words = list(bi_cands.keys())
            probs = np.array(list(bi_cands.values()), dtype=float)
            probs /= probs.sum()
            return str(_rng.choice(words, p=probs))
        # 유니그램 폴백
        if self.uni:
            top = [w for w,_ in self.uni.most_common(50)
                   if w not in exclude]
            if top: return top[0]
        return ""

    def compose_from_sentences(self, sentences: list,
                               query_info: dict,
                               max_len: int = 25) -> str:
        """
        관련 문장들에서 답변 조합
        질문 유형별 전략
        """
        if not sentences: return ""
        q_type = query_info["type"]

        # 수치/시간 질문: 관련 문장 직접 반환
        if q_type in ("수치", "시간"):
            best = sentences[0][0]
            return best

        # 목록 질문: 여러 문장 결합
        if q_type == "목록" and len(sentences) >= 2:
            parts = []
            for s, _ in sentences[:3]:
                toks = tokenize(s)
                if toks: parts.extend(toks[:5])
            if parts:
                # 중복 제거 + 조합
                seen = set()
                unique = [t for t in parts
                          if t not in seen and not seen.add(t)]
                return " ".join(unique[:15])

        # 여부 질문: 가장 관련 높은 문장 + 요약
        if q_type == "여부":
            best = sentences[0][0]
            keywords = query_info["keywords"]
            found = any(k in best for k in keywords)
            if found:
                return best
            return best

        # 일반/기타: 마르코프로 생성
        if len(tokenize(" ".join(s for s,_ in sentences[:2]))) > 3:
            seed_toks = tokenize(sentences[0][0])[:3]
            return self._generate_from_seed(seed_toks, max_len)

        return sentences[0][0]

    def _generate_from_seed(self, seed: list, max_len: int) -> str:
        if not seed or self.total == 0: return ""
        result = list(seed)
        used   = set(seed)
        for _ in range(max_len - len(seed)):
            prev  = result[-1]
            prev2 = result[-2] if len(result)>=2 else None
            nxt   = self.next_token(prev, prev2,
                                    exclude=set(result[-2:]))
            if not nxt: break
            result.append(nxt)
            # 자연스러운 종결 감지
            if any(result[-1].endswith(e)
                   for e in ["다","요","음","임","됨","됩니다","합니다"]):
                if len(result) >= 5: break
        return " ".join(result)


# ── NeuralMarkov 검증기 ───────────────────────────────────────
class NMVerifier:
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
        scores = [self._s(base[i:i+5])
                  for i in range(0,len(base)-5,3)]
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

    def score(self, text) -> dict:
        s = self._s(tokenize(text))
        st = "PASS" if s>=-10 else "WARNING" if s>=-13 else "FATAL"
        return {"status":st,"logp":s,"z":(s-self.mu)/self.std}


# ── 통합 생성 엔진 ────────────────────────────────────────────
class OfflineGenerationEngine:
    """
    완전 오프라인 답변 생성 엔진
    LLM 없이 독립 작동
    """
    def __init__(self):
        self.analyzer  = QueryAnalyzer()
        self.retriever = SentenceRetriever()
        self.generator = MarkovGenerator()
        self.verifier  = NMVerifier()
        self.polar_engine = None
        self.is_ready  = False
        self.corpus    = ""

    def build(self, corpus_text: str, polar_engine=None,
              epochs: int = 20):
        self.corpus = corpus_text
        self.polar_engine = polar_engine
        self.retriever.build(corpus_text, polar_engine)
        self.generator.train(corpus_text, epochs=epochs)
        self.verifier.train(corpus_text, epochs=epochs)
        self.is_ready = True

    def generate(self, question: str,
                 topk: int = 5) -> dict:
        t0 = time.perf_counter()

        # 1. 질문 분석
        q_info = self.analyzer.analyze(question)

        # 2. 관련 문장 검색
        related = self.retriever.search(
            q_info, self.polar_engine, topk=topk)

        # 3. 답변 생성
        if related:
            answer = self.generator.compose_from_sentences(
                related, q_info, max_len=30)
        else:
            answer = ""

        # 4. 품질 검증
        if answer:
            quality = self.verifier.score(answer)
        else:
            quality = {"status":"FATAL","logp":-20.0,"z":-5.0}

        ms = (time.perf_counter()-t0)*1000
        return {
            "answer":    answer,
            "quality":   quality,
            "q_type":    q_info["type"],
            "keywords":  q_info["keywords"],
            "related":   [(s[:50],round(sc,3)) for s,sc in related[:3]],
            "ms":        ms,
            "offline":   True,
        }
