"""
online_collector.py — 온라인 코퍼스 수집기
============================================
온라인: URL/RSS/검색으로 코퍼스 수집
오프라인: 수집된 코퍼스로 극성 임베딩 학습
→ pkl 저장 → 완전 오프라인 전환

수집 소스:
  1. URL 직접 입력 (웹페이지 텍스트 추출)
  2. RSS 피드 (뉴스/블로그)
  3. 키워드 검색 (DuckDuckGo)
  4. 파일 업로드 (txt/pdf/docx)
"""
from __future__ import annotations
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field


# ── 수집 결과 ─────────────────────────────────────────────────
@dataclass
class CollectedDoc:
    source:     str           # URL or keyword
    title:      str = ""
    text:       str = ""
    tokens:     int = 0
    sentences:  int = 0
    collected_at: float = field(default_factory=time.time)
    status:     str = "ok"   # ok / error / empty
    error:      str = ""


# ── 텍스트 클리너 ─────────────────────────────────────────────
class TextCleaner:
    """HTML/불필요 문자 제거 + 문장 분리"""

    # 제거 패턴
    _HTML_TAG    = re.compile(r'<[^>]+>')
    _MULTI_SPACE = re.compile(r'\s+')
    _SCRIPT      = re.compile(r'<script[^>]*>.*?</script>',re.DOTALL|re.IGNORECASE)
    _STYLE       = re.compile(r'<style[^>]*>.*?</style>',  re.DOTALL|re.IGNORECASE)
    _HTML_ENT    = re.compile(r'&[a-zA-Z]+;|&#[0-9]+;')

    def clean_html(self, html: str) -> str:
        text = self._SCRIPT.sub('', html)
        text = self._STYLE.sub('', text)
        text = self._HTML_TAG.sub(' ', text)
        text = self._HTML_ENT.sub(' ', text)
        text = self._MULTI_SPACE.sub(' ', text)
        return text.strip()

    def extract_sentences(self, text: str,
                          min_len: int = 10,
                          max_len: int = 300) -> list:
        """의미 있는 문장만 추출"""
        # 문장 분리
        sents = re.split(r'[.!?\n。]', text)
        result = []
        for s in sents:
            s = s.strip()
            if min_len <= len(s) <= max_len:
                # 한국어 또는 영어 문자 포함 확인
                if re.search(r'[가-힣a-zA-Z]', s):
                    result.append(s)
        return result

    def to_corpus(self, sentences: list) -> str:
        return "\n".join(sentences)


# ── URL 수집기 ────────────────────────────────────────────────
class URLCollector:
    """단일 URL에서 텍스트 수집"""

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.cleaner = TextCleaner()

    def collect(self, url: str) -> CollectedDoc:
        doc = CollectedDoc(source=url)
        try:
            req = urllib.request.Request(url, headers=self.HEADERS)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                # 인코딩 감지
                encoding = resp.headers.get_content_charset() or 'utf-8'
                try:
                    html = raw.decode(encoding, errors='ignore')
                except Exception:
                    html = raw.decode('utf-8', errors='ignore')

            # 제목 추출
            title_m = re.search(r'<title[^>]*>(.*?)</title>',
                                 html, re.IGNORECASE|re.DOTALL)
            if title_m:
                doc.title = self.cleaner.clean_html(title_m.group(1))[:80]

            text       = self.cleaner.clean_html(html)
            sentences  = self.cleaner.extract_sentences(text)
            doc.text   = self.cleaner.to_corpus(sentences)
            doc.tokens = len(doc.text.split())
            doc.sentences = len(sentences)

            if doc.tokens < 20:
                doc.status = "empty"
                doc.error  = "텍스트가 너무 짧아요"
        except urllib.error.HTTPError as e:
            doc.status = "error"; doc.error = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            doc.status = "error"; doc.error = f"연결 실패: {e.reason}"
        except Exception as e:
            doc.status = "error"; doc.error = str(e)[:80]
        return doc


# ── RSS 수집기 ────────────────────────────────────────────────
class RSSCollector:
    """RSS 피드에서 뉴스/블로그 수집"""

    # Google News RSS (API 불필요)
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    # Naver 검색 RSS (API 키 필요 시 선택)
    NAVER_NEWS_RSS  = "https://openapi.naver.com/v1/search/news.xml?query={query}&display=10"

    DEFAULT_FEEDS = {
        # ── 한국 종합 ──────────────────────────────────────
        "연합뉴스 전체":
            "https://www.yonhapnews.co.kr/RSS/headline.xml",
        "연합뉴스 교육":
            "https://www.yonhapnews.co.kr/RSS/education.xml",
        "연합뉴스 IT/과학":
            "https://www.yonhapnews.co.kr/RSS/it.xml",
        "연합뉴스 사회":
            "https://www.yonhapnews.co.kr/RSS/society.xml",
        # ── 나무위키 ────────────────────────────────────────
        "나무위키 최근변경":
            "https://namu.wiki/atom",
        # ── 네이버 뉴스 섹션 ───────────────────────────────
        "네이버 사회":
            "https://news.naver.com/rss/section_102.xml",
        "네이버 IT과학":
            "https://news.naver.com/rss/section_105.xml",
        "네이버 경제":
            "https://news.naver.com/rss/section_101.xml",
        "네이버 교육":
            "https://news.naver.com/rss/section_083.xml",
        "네이버 세계":
            "https://news.naver.com/rss/section_104.xml",
        # ── Google 뉴스 (키워드) ───────────────────────────
        "Google 뉴스 AI":
            "https://news.google.com/rss/search?q=인공지능&hl=ko&gl=KR&ceid=KR:ko",
        "Google 뉴스 교육":
            "https://news.google.com/rss/search?q=교육&hl=ko&gl=KR&ceid=KR:ko",
        "Google 뉴스 과학":
            "https://news.google.com/rss/search?q=과학기술&hl=ko&gl=KR&ceid=KR:ko",
        "Google 뉴스 사회":
            "https://news.google.com/rss/search?q=사회&hl=ko&gl=KR&ceid=KR:ko",
        # ── 위키백과 ────────────────────────────────────────
        "위키백과 최근변경":
            "https://ko.wikipedia.org/w/api.php?action=feedrecentchanges&lang=ko&feedformat=rss",
        # ── 영어 (글로벌) ──────────────────────────────────
        "BBC News":
            "http://feeds.bbci.co.uk/news/rss.xml",
        "Reuters Top News":
            "https://feeds.reuters.com/reuters/topNews",
        "Hacker News":
            "https://news.ycombinator.com/rss",
        "MIT Tech Review":
            "https://www.technologyreview.com/feed/",
    }

    def search_google_news(self, keyword: str,
                           max_items: int = 5) -> list:
        """Google 뉴스 RSS 검색 (API 불필요)"""
        import urllib.parse
        query   = urllib.parse.quote(keyword)
        feed_url= self.GOOGLE_NEWS_RSS.format(query=query)
        return self.collect_feed(feed_url, max_items)

    def search_naver_news(self, keyword: str,
                          client_id: str = "",
                          client_secret: str = "",
                          max_items: int = 5) -> list:
        """네이버 뉴스 검색 (API 키 없으면 RSS 폴백)"""
        if client_id and client_secret:
            # 네이버 검색 API
            import urllib.parse, urllib.request
            query = urllib.parse.quote(keyword)
            url   = self.NAVER_NEWS_RSS.format(query=query)
            req   = urllib.request.Request(url, headers={
                "X-Naver-Client-Id":     client_id,
                "X-Naver-Client-Secret": client_secret,
                "User-Agent": URLCollector.HEADERS["User-Agent"],
            })
            try:
                import re
                with urllib.request.urlopen(req, timeout=10) as resp:
                    xml = resp.read().decode("utf-8", errors="ignore")
                links  = re.findall(r'<link[^>]*>([^<]+)</link>', xml)
                titles = re.findall(r'<title[^>]*><!\[CDATA\[([^\]]+)\]\]></title>', xml)
                docs = []
                for i, link in enumerate(links[:max_items]):
                    link = link.strip()
                    if not link.startswith("http"): continue
                    doc = self.url_col.collect(link)
                    if i < len(titles): doc.title = titles[i][:80]
                    if doc.status == "ok" and doc.tokens > 30:
                        docs.append(doc)
                    import time; time.sleep(0.3)
                return docs
            except Exception as e:
                pass
        # 폴백: 네이버 뉴스 섹션 RSS
        return self.collect_feed(
            f"https://news.naver.com/rss/section_083.xml", max_items)

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.cleaner = TextCleaner()
        self.url_col = URLCollector(timeout)

    def collect_feed(self, feed_url: str,
                     max_items: int = 10) -> list[CollectedDoc]:
        """RSS 피드에서 기사 목록 수집"""
        docs = []
        try:
            req = urllib.request.Request(
                feed_url,
                headers=URLCollector.HEADERS)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                xml = resp.read().decode('utf-8', errors='ignore')

            # item/link 파싱
            links = re.findall(r'<link[^>]*>([^<]+)</link>', xml)
            titles= re.findall(r'<title[^>]*><!\[CDATA\[([^\]]+)\]\]></title>',xml)
            if not titles:
                titles = re.findall(r'<title[^>]*>([^<]+)</title>', xml)

            for i, link in enumerate(links[:max_items]):
                link = link.strip()
                if not link.startswith('http'): continue
                doc = self.url_col.collect(link)
                if i < len(titles):
                    doc.title = titles[i][:80]
                if doc.status == "ok" and doc.tokens > 30:
                    docs.append(doc)
                time.sleep(0.3)  # 서버 부하 방지
        except Exception as e:
            docs.append(CollectedDoc(
                source=feed_url, status="error", error=str(e)[:80]))
        return docs


# ── 키워드 검색 수집기 ────────────────────────────────────────
class SearchCollector:
    """
    DuckDuckGo HTML 검색으로 관련 URL 수집
    (API 키 불필요)
    """
    DDGO_URL = "https://html.duckduckgo.com/html/?q={query}"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.url_col = URLCollector(timeout)
        self.cleaner = TextCleaner()

    def search_urls(self, keyword: str, max_results: int = 5) -> list:
        """DuckDuckGo에서 URL 목록 수집"""
        try:
            query   = urllib.parse.quote(keyword)
            url     = self.DDGO_URL.format(query=query)
            req     = urllib.request.Request(url, headers=URLCollector.HEADERS)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode('utf-8', errors='ignore')

            # 결과 URL 파싱
            urls = re.findall(
                r'<a[^>]+class="result__url"[^>]*>([^<]+)</a>', html)
            if not urls:
                urls = re.findall(
                    r'href="//duckduckgo\.com/l/\?uddg=([^"]+)"', html)
                urls = [urllib.parse.unquote(u) for u in urls]

            # http 없는 것 보완
            result = []
            for u in urls:
                u = u.strip()
                if not u.startswith('http'):
                    u = 'https://' + u
                if u.startswith('http'):
                    result.append(u)
            return result[:max_results]
        except Exception:
            return []

    def collect(self, keyword: str,
                max_results: int = 3) -> list[CollectedDoc]:
        """키워드로 검색 후 상위 페이지 수집"""
        urls = self.search_urls(keyword, max_results)
        docs = []
        for url in urls:
            doc = self.url_col.collect(url)
            if doc.status == "ok" and doc.tokens > 50:
                docs.append(doc)
            time.sleep(0.5)
        return docs


# ── 수집 파이프라인 ───────────────────────────────────────────
class CollectionPipeline:
    """
    수집 → 정제 → 학습 통합 파이프라인
    """
    def __init__(self):
        self.url_collector    = URLCollector()
        self.rss_collector    = RSSCollector()
        self.search_collector = SearchCollector()
        self.cleaner          = TextCleaner()
        self.collected_docs:  list[CollectedDoc] = []
        self.total_tokens     = 0

    def collect_url(self, url: str) -> CollectedDoc:
        doc = self.url_collector.collect(url)
        if doc.status == "ok":
            self.collected_docs.append(doc)
            self.total_tokens += doc.tokens
        return doc

    def collect_rss(self, feed_url: str,
                    max_items: int = 5) -> list[CollectedDoc]:
        docs = self.rss_collector.collect_feed(feed_url, max_items)
        for d in docs:
            if d.status == "ok":
                self.collected_docs.append(d)
                self.total_tokens += d.tokens
        return docs

    def collect_keyword(self, keyword: str,
                        max_results: int = 3) -> list[CollectedDoc]:
        docs = self.search_collector.collect(keyword, max_results)
        for d in docs:
            if d.status == "ok":
                self.collected_docs.append(d)
                self.total_tokens += d.tokens
        return docs

    def collect_text(self, text: str,
                     source: str = "직접입력") -> CollectedDoc:
        """직접 입력 텍스트를 도큐멘트로 추가"""
        sentences = self.cleaner.extract_sentences(text)
        corpus    = self.cleaner.to_corpus(sentences)
        doc = CollectedDoc(
            source=source,
            title="직접 입력",
            text=corpus,
            tokens=len(corpus.split()),
            sentences=len(sentences),
        )
        if doc.tokens > 5:
            self.collected_docs.append(doc)
            self.total_tokens += doc.tokens
        return doc

    def get_merged_corpus(self) -> str:
        """수집된 모든 문서를 하나의 코퍼스로 합치기"""
        parts = []
        for doc in self.collected_docs:
            if doc.status == "ok" and doc.text:
                parts.append(doc.text)
        return "\n".join(parts)

    def train_on_collected(self, engine,
                           polar_engine=None,
                           epochs: int = 10,
                           on_progress=None) -> dict:
        """수집된 코퍼스로 엔진 학습"""
        corpus = self.get_merged_corpus()
        if not corpus.strip():
            return {"error": "수집된 텍스트가 없어요"}

        if on_progress: on_progress(20, "코퍼스 병합 완료...")

        # 오프라인 생성 엔진 학습
        t0 = time.perf_counter()
        engine.build(corpus, polar_engine=polar_engine, epochs=epochs)
        engine_ms = (time.perf_counter()-t0)*1000

        if on_progress: on_progress(70, "극성 임베딩 학습...")

        # 극성 임베딩 학습
        if polar_engine:
            polar_engine.train_corpus(corpus, window=4, epochs=min(epochs,5))

        if on_progress: on_progress(100, "완료")
        train_ms = (time.perf_counter()-t0)*1000

        return {
            "corpus_len":   len(corpus),
            "total_tokens": self.total_tokens,
            "n_docs":       len(self.collected_docs),
            "train_ms":     train_ms,
        }

    def clear(self):
        self.collected_docs = []
        self.total_tokens   = 0

    def summary(self) -> dict:
        ok_docs = [d for d in self.collected_docs if d.status=="ok"]
        return {
            "전체 문서": len(self.collected_docs),
            "성공":      len(ok_docs),
            "실패":      len(self.collected_docs)-len(ok_docs),
            "총 토큰":   self.total_tokens,
            "총 문장":   sum(d.sentences for d in ok_docs),
        }
