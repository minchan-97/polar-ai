"""
app.py — Polar AI 통합 앱 (모바일 최적화)
==========================================
Polar-SSM + NM 가드레일 + XAI + 메모리
한 화면에 전부 표시 / 모바일 호환 PKL
"""
import streamlit as st
import io, os, time, pickle
import numpy as np

st.set_page_config(
    page_title="Polar AI",
    page_icon="🧠",
    layout="centered",   # 모바일: centered가 최적
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
:root{
  --bg:#0a0e1a;--sf:#111827;--bd:#1e3050;
  --ac:#00e5ff;--gn:#00ff88;--yw:#ffd600;--rd:#ff4444;
  --tx:#e2e8f0;--mt:#64748b;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--tx)!important;
  font-family:'Noto Sans KR',sans-serif;
}
/* 모바일 최적화 */
[data-testid="stAppViewContainer"]{padding:0.5rem!important;}
[data-testid="stVerticalBlock"]{gap:0.4rem!important;}

.card{background:var(--sf);border:1px solid var(--bd);
  border-radius:10px;padding:0.9rem;margin-bottom:0.5rem;}
.badge{display:inline-block;padding:2px 9px;border-radius:4px;
  font-family:'Space Mono',monospace;font-size:0.6rem;font-weight:700;
  letter-spacing:0.5px;}
.b-pass{background:#0d2a0d;border:1px solid #00ff88;color:#00ff88;}
.b-warn{background:#2a1f0d;border:1px solid #ffd600;color:#ffd600;}
.b-fatal{background:#2a0d0d;border:1px solid #ff4444;color:#ff4444;}
.b-ssm{background:#0d1a2a;border:1px solid #00e5ff;color:#00e5ff;}
.b-fb{background:#1a1a0d;border:1px solid #ffd600;color:#ffd600;}

/* 채팅 말풍선 */
.bubble-user{
  background:#1e3a5f;color:#e2e8f0;
  padding:8px 14px;border-radius:14px 14px 4px 14px;
  display:inline-block;max-width:85%;font-size:0.9rem;
  margin:3px 0;float:right;clear:both;
}
.bubble-ai{
  background:#111827;color:#e2e8f0;
  border:1px solid var(--bd);
  padding:8px 14px;border-radius:14px 14px 14px 4px;
  display:inline-block;max-width:85%;font-size:0.9rem;
  margin:3px 0;float:left;clear:both;
}
.xai-box{
  background:#0a0f1a;border:1px solid #1e3050;border-radius:6px;
  padding:5px 10px;margin-top:4px;font-size:0.72rem;color:#64748b;
  font-family:'Space Mono',monospace;
}
.tok-p{color:#00ff88;} .tok-w{color:#ffd600;} .tok-f{color:#ff4444;}
.clearfix{clear:both;height:4px;}

/* 버튼 */
[data-testid="stButton"] button{
  width:100%!important;font-weight:700!important;
  border-radius:8px!important;border:none!important;
  font-size:0.9rem!important;padding:0.6rem!important;
}
/* 입력창 */
[data-testid="stTextArea"] textarea{
  font-size:0.95rem!important;border-radius:8px!important;
}
/* 슬라이더 */
[data-testid="stSlider"]{padding:0!important;}
hr{border-color:var(--bd)!important;}
/* expander */
[data-testid="stExpander"]{
  background:var(--sf)!important;border:1px solid var(--bd)!important;
  border-radius:8px!important;
}
</style>""", unsafe_allow_html=True)

try:
    from core import PolarAICore
    from online_collector import CollectionPipeline, RSSCollector
    from auto_train import SOMBuilder, AutoTrainer
    CORE_OK = True
except Exception as e:
    CORE_OK = False
    st.error(f"코어 오류: {e}")
    st.stop()

# ── 세션 ─────────────────────────────────────────────────────
for k,v in {
    "core":         None,
    "ready":        False,
    "corpus_bytes": None,
    "corpus_name":  "",
    "pkl_bytes":    None,
    "logp_thr":     -11.5,
    "epochs":       20,
    "collector":    None,
    "som_builder":  None,
    "trainer":      None,
    "training_active": False,
    "naver_id":     "",
    "naver_secret": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_core() -> PolarAICore:
    if st.session_state.core is None:
        st.session_state.core = PolarAICore()
    return st.session_state.core


# ━━━ 헤더 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div style="text-align:center;padding:0.5rem 0 0.8rem;">
  <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
  font-weight:700;color:#00e5ff;">🧠 Polar AI</div>
  <div style="font-size:0.7rem;color:#64748b;margin-top:2px;">
  Polar-SSM · NM 가드레일 · XAI · 오프라인 완전 독립
  </div>
</div>""", unsafe_allow_html=True)


# ━━━ 섹션 1: 학습 / 로드 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander(
    "⚙️ 학습" +
    (" ✅" if st.session_state.ready else " — 학습 방법을 선택하세요"),
    expanded=not st.session_state.ready,
):
    # 학습 모드 선택
    mode = st.radio(
        "학습 방법",
        ["📄 파일 코퍼스 (PDF/TXT/DOCX)",
         "🌐 온라인 수집 코퍼스",
         "💾 저장된 AI 불러오기"],
        key="train_mode",
        label_visibility="collapsed",
    )

    # ── 모드 1: 파일 ────────────────────────────────────────
    if mode == "📄 파일 코퍼스 (PDF/TXT/DOCX)":
        corp_up = st.file_uploader(
            "파일 업로드", type=["txt","pdf","docx"],
            key="corp_up", label_visibility="collapsed")
        if corp_up:
            st.session_state.corpus_bytes = corp_up.read()
            st.session_state.corpus_name  = corp_up.name

        corp_text = st.text_area(
            "또는 직접 입력", height=80,
            placeholder="학습할 텍스트...",
            label_visibility="collapsed", key="corp_text")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.epochs = st.select_slider(
                "epochs", [5,10,15,20],
                value=st.session_state.epochs)
        with c2:
            st.session_state.logp_thr = st.select_slider(
                "임계값", [-15.0,-13.0,-11.5,-10.0,-8.0],
                value=st.session_state.logp_thr)

        if st.button("🚀 파일로 학습", use_container_width=True):
            raw  = st.session_state.corpus_bytes
            name = st.session_state.corpus_name.lower()
            text = corp_text.strip()
            if raw and not text:
                try:
                    if name.endswith(".pdf"):
                        import pypdf
                        text = "\n".join(
                            p.extract_text() or ""
                            for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
                    elif name.endswith(".docx"):
                        import docx
                        text = "\n".join(
                            p.text for p in
                            docx.Document(io.BytesIO(raw)).paragraphs
                            if p.text.strip())
                    else:
                        text = raw.decode("utf-8", errors="ignore")
                except Exception as e:
                    st.error(f"파일 읽기 실패: {e}"); text = ""
            if not text:
                st.warning("파일을 업로드하거나 텍스트를 입력하세요.")
            else:
                prog = st.progress(0)
                def cb(pct, msg): prog.progress(pct)
                with st.spinner("학습 중..."):
                    try:
                        core = PolarAICore()
                        core.build(text,
                                   epochs=st.session_state.epochs,
                                   on_progress=cb)
                        st.session_state.core  = core
                        st.session_state.ready = True
                        st.success("✓ 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"학습 실패: {e}")

    # ── 모드 2: 온라인 수집 ─────────────────────────────────
    elif mode == "🌐 온라인 수집 코퍼스":
        if st.session_state.collector is None:
            st.session_state.collector = CollectionPipeline()
        col = st.session_state.collector

        src = st.radio("수집 방법", [
            "🔍 Google 뉴스",
            "📰 네이버 뉴스",
            "📡 RSS 피드",
            "🔗 URL 직접",
            "✍️ 텍스트",
        ], key="src_mode2", label_visibility="collapsed",
           horizontal=True)

        if src == "🔍 Google 뉴스":
            gkw = st.text_input("검색어", key="gkw2",
                                placeholder="검색어 입력 (제한 없음)")
            g_max = st.slider("최대 기사", 3, 15, 5, key="g_max2")
            if st.button("🔍 수집", use_container_width=True):
                if gkw:
                    with st.spinner(f"'{gkw}' Google 뉴스 수집..."):
                        try:
                            rss = RSSCollector()
                            docs = rss.search_google_news(gkw, g_max)
                            ok = [d for d in docs if d.status=="ok"]
                            for d in ok:
                                col.collected_docs.append(d)
                                col.total_tokens += d.tokens
                            st.success(f"✓ {len(ok)}개 ({sum(d.tokens for d in ok)}토큰)")
                        except Exception as e:
                            st.error(f"실패: {e}")
                else:
                    st.warning("검색어를 입력하세요.")

        elif src == "📰 네이버 뉴스":
            nkw = st.text_input("검색어", key="nkw2",
                                placeholder="검색어 (제한 없음)")
            with st.expander("API 설정 (선택)", expanded=False):
                st.session_state.naver_id = st.text_input(
                    "Client ID", value=st.session_state.naver_id)
                st.session_state.naver_secret = st.text_input(
                    "Client Secret",
                    value=st.session_state.naver_secret,
                    type="password")
                st.caption("없으면 네이버 섹션 RSS 사용")
            n_max2 = st.slider("최대 기사", 3, 15, 5, key="n_max2")
            if st.button("📰 수집", use_container_width=True):
                if nkw:
                    with st.spinner(f"'{nkw}' 네이버 수집..."):
                        try:
                            rss = RSSCollector()
                            docs = rss.search_naver_news(
                                nkw,
                                client_id=st.session_state.naver_id,
                                client_secret=st.session_state.naver_secret,
                                max_items=n_max2)
                            ok = [d for d in docs if d.status=="ok"]
                            for d in ok:
                                col.collected_docs.append(d)
                                col.total_tokens += d.tokens
                            st.success(f"✓ {len(ok)}개")
                        except Exception as e:
                            st.error(f"실패: {e}")

        elif src == "📡 RSS 피드":
            from online_collector import RSSCollector as _RC
            _presets = _RC.DEFAULT_FEEDS if hasattr(_RC,'DEFAULT_FEEDS') else {}
            preset_list = ["직접 입력"] + list(
                RSSCollector.DEFAULT_FEEDS.keys())
            sel = st.selectbox("프리셋", preset_list, key="rss_sel2")
            rss_url2 = st.text_input(
                "RSS URL",
                value=RSSCollector.DEFAULT_FEEDS.get(sel,"")
                      if sel != "직접 입력" else "",
                key="rss_url2")
            r_max2 = st.slider("최대 기사", 3, 20, 8, key="r_max2")
            if st.button("📡 RSS 수집", use_container_width=True):
                if rss_url2:
                    with st.spinner("RSS 수집 중..."):
                        try:
                            rss_col = RSSCollector()
                            docs = rss_col.collect_feed(rss_url2, r_max2)
                            ok = [d for d in docs if d.status=="ok"]
                            for d in ok:
                                col.collected_docs.append(d)
                                col.total_tokens += d.tokens
                            st.success(f"✓ {len(ok)}/{len(docs)}개")
                        except Exception as e:
                            st.error(f"실패: {e}")

        elif src == "🔗 URL 직접":
            urls_txt = st.text_area("URL 목록", height=80,
                                    key="urls2",
                                    placeholder="https://...")
            if st.button("📥 수집", use_container_width=True):
                urls = [u.strip() for u in urls_txt.split("\n")
                        if u.strip().startswith("http")]
                if urls:
                    prog2 = st.progress(0)
                    for i, url in enumerate(urls):
                        doc = col.collect_url(url)
                        prog2.progress((i+1)/len(urls))
                        st.caption(f"{'✓' if doc.status=='ok' else '✗'} {url[:40]}")
                else:
                    st.warning("URL을 입력하세요.")

        elif src == "✍️ 텍스트":
            dtxt = st.text_area("텍스트", height=100, key="dtxt2",
                                placeholder="직접 입력...")
            if st.button("➕ 추가", use_container_width=True):
                if dtxt.strip():
                    doc = col.collect_text(dtxt)
                    st.success(f"✓ {doc.tokens}토큰")

        # 수집 현황
        s_col = col.summary()
        c1,c2 = st.columns(2)
        c1.metric("수집 문서", s_col["성공"])
        c2.metric("총 토큰",   s_col["총 토큰"])

        if col.collected_docs:
            with st.expander(f"목록 ({len(col.collected_docs)}개)"):
                for d in col.collected_docs[-10:]:
                    st.caption(
                        f"{'✅' if d.status=='ok' else '❌'} "
                        f"{d.title[:40]} ({d.tokens}토큰)")
            ca, cb2 = st.columns(2)
            with ca:
                ep2 = st.select_slider("epochs", [5,10,15,20],
                                       value=10, key="ep2")
            with cb2:
                st.session_state.logp_thr = st.select_slider(
                    "임계값",
                    [-15.0,-13.0,-11.5,-10.0,-8.0],
                    value=st.session_state.logp_thr,
                    key="thr2")
            c1,c2 = st.columns(2)
            with c1:
                if st.button("🚀 수집 내용으로 학습",
                             use_container_width=True):
                    corpus = col.get_merged_corpus()
                    if corpus.strip():
                        prog3 = st.progress(0)
                        def cb3(p,m): prog3.progress(p)
                        with st.spinner("학습 중..."):
                            core2 = PolarAICore()
                            core2.build(corpus, epochs=ep2,
                                        on_progress=cb3)
                            st.session_state.core  = core2
                            st.session_state.ready = True
                        st.success("✓ 완료!")
                        st.rerun()
            with c2:
                if st.button("🗑️ 초기화",
                             use_container_width=True,
                             key="col_clr2"):
                    col.clear(); st.rerun()

    # ── 모드 3: PKL 불러오기 ─────────────────────────────────
    else:
        pkl_up = st.file_uploader(
            "PKL 파일", type=["pkl"], key="pkl_up",
            label_visibility="collapsed")
        if pkl_up:
            st.session_state.pkl_bytes = pkl_up.read()

        if st.session_state.pkl_bytes:
            if st.button("📂 AI 불러오기",
                         use_container_width=True, key="load2"):
                with st.spinner("복원 중..."):
                    try:
                        core = PolarAICore.from_bytes(
                            st.session_state.pkl_bytes)
                        st.session_state.core  = core
                        st.session_state.ready = True
                        s2 = core.summary()
                        st.success(
                            f"✓ {s2['극성 어휘']}어휘 | "
                            f"대화 {s2['전체 대화']}턴")
                        st.rerun()
                    except Exception as e:
                        st.error(f"실패: {e}")

# ━━━ 학습 전 스탑 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if not st.session_state.ready:
    st.info("위 설정에서 코퍼스를 학습하거나 저장된 AI를 불러오세요.")
    st.stop()

core = get_core()


# ━━━ 섹션 2: 상태 바 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
s = core.summary()
c1,c2,c3,c4 = st.columns(4)
c1.metric("어휘",    s["극성 어휘"])
c2.metric("문장",    s["문장 인덱스"])
c3.metric("대화",    s["전체 대화"])
c4.metric("세션",    s["대화 세션"])


# ━━━ 섹션 3: 채팅 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")

# 세션 없으면 시작
if not core.memory.current_id:
    core.memory.new_session()

sess = core.memory.get_current()

# 대화 표시
chat_area = st.container()
with chat_area:
    if sess and sess.turns:
        turns = sess.turns[-16:]  # 최근 16개 (모바일 성능)
        if sess.summary:
            st.caption(f"📝 {sess.summary}")

        for i, turn in enumerate(turns):
            if turn.role == "user":
                st.markdown(
                    f'<div class="bubble-user">{turn.content}</div>'
                    f'<div class="clearfix"></div>',
                    unsafe_allow_html=True)
            else:
                method_cls = {"ssm_pass":"b-ssm","ssm_warning":"b-warn",
                              "fallback":"b-fb","none":"b-fatal"}.get(
                              turn.source, "b-ssm")
                method_lbl = {"ssm_pass":"SSM✅","ssm_warning":"SSM⚠️",
                              "fallback":"폴백","none":"없음"}.get(
                              turn.source, turn.source)
                qual_cls   = {"PASS":"b-pass","WARNING":"b-warn",
                              "FATAL":"b-fatal"}.get(turn.quality,"")
                st.markdown(
                    f'<div class="bubble-ai">'
                    f'<span class="badge {qual_cls}">'
                    f'{turn.quality or "?"}</span> '
                    f'<span class="badge {method_cls}">{method_lbl}</span>'
                    f'<br>{turn.content}</div>'
                    f'<div class="clearfix"></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;color:#64748b;'
            'padding:1.5rem;font-size:0.85rem;">대화를 시작하세요</div>',
            unsafe_allow_html=True)


# ━━━ 섹션 4: 입력창 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
msg = st.text_area(
    "메시지",
    height=68,
    placeholder="질문을 입력하세요...",
    label_visibility="collapsed",
    key="msg_input",
)
ca, cb = st.columns([3,1])
with ca:
    send = st.button("📤 전송", key="send_btn",
                     disabled=not msg.strip())
with cb:
    new_sess = st.button("🆕", key="new_sess",
                         help="새 대화")

if new_sess:
    core.memory.new_session()
    st.rerun()

if send and msg.strip():
    with st.spinner("생각 중..."):
        result = core.answer(
            msg.strip(),
            logp_thr=st.session_state.logp_thr,
            use_memory=True,
        )
    # XAI 접기
    xai = result.get("xai", {})
    if xai:
        xai_html = (
            f'<div class="xai-box">'
            f'🔬 {xai.get("why","")} | '
            f'🧭 {xai.get("state_desc","")}'
            f'</div>'
        )
        st.session_state._last_xai = xai_html
    st.rerun()

# 마지막 XAI 표시
if hasattr(st.session_state, "_last_xai") and st.session_state._last_xai:
    st.markdown(st.session_state._last_xai, unsafe_allow_html=True)
    st.session_state._last_xai = ""


# ━━━ 섹션 5: SOM 자동학습 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("🤖 SOM 자동학습", expanded=False):
    if not st.session_state.ready:
        st.info("먼저 코퍼스를 학습하세요.")
    else:
        core = get_core()
        st.caption("SOM 의미 지도 → 클러스터 선택 → 시간 설정 → 자동 반복 학습")

        dur_min = st.slider("학습 시간(분)", 1, 20, 5, key="auto_dur")
        api_at  = st.text_input("OpenAI Key (선택)", type="password",
                                key="auto_key",
                                help="없으면 패턴 기반 자동 생성")
        ca, cb_som = st.columns(2)
        with ca:
            if st.button("🗺️ SOM 빌드", use_container_width=True,
                         disabled=len(core.polar.word_vecs)<5):
                with st.spinner("SOM 빌드 중..."):
                    sb = SOMBuilder(grid=6)
                    ok = sb.build_from_engine(core.polar)
                if ok:
                    st.session_state.som_builder = sb
                    st.success(f"✓ {len(sb.neuron_data)}클러스터")
                else:
                    st.warning("어휘 부족")

        with cb_som:
            sb = st.session_state.som_builder
            if sb and sb.neuron_data:
                sel_n = st.number_input(
                    "뉴런 번호", 0, sb.grid*sb.grid-1, 0,
                    key="sel_n_auto")
                if sel_n in sb.neuron_data:
                    words = [it["word"]
                             for it in sb.neuron_data[sel_n][:4]]
                    st.caption(f"📍 {', '.join(words)}")

        sb = st.session_state.som_builder
        if sb and sb.neuron_data and not st.session_state.training_active:
            if st.button(f"⚡ 자동학습 시작 ({dur_min}분)",
                         use_container_width=True):
                st.session_state.training_active = True
                sel_n = st.session_state.get("sel_n_auto", 0)
                trainer = AutoTrainer(core.polar, sb,
                                      duration_sec=dur_min*60)
                st.session_state.trainer = trainer
                trainer.start_time = time.time()

                prog_a = st.progress(0)
                stat_a = st.empty()
                while trainer.remaining() > 0:
                    r = trainer.train_round(
                        sel_n, api_key=api_at, epochs_per_round=2)
                    prog_a.progress(min(trainer.progress(),1.0))
                    stat_a.caption(
                        f"라운드 {trainer.rounds_done} | "
                        f"+{r['new_words']}어휘 | "
                        f"남은 {trainer.remaining():.0f}초")
                    if trainer.rounds_done % 5 == 0:
                        sb.build_from_engine(core.polar)

                st.session_state.training_active = False
                st.success(
                    f"✅ {trainer.rounds_done}라운드 완료 | "
                    f"+{trainer.words_added}어휘")
                st.rerun()


# ━━━ 섹션 6: 저장 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # collector 초기화
    if st.session_state.collector is None:
        st.session_state.collector = CollectionPipeline()
    col = st.session_state.collector

    st.markdown("**수집 방법 선택**")
    src = st.radio("", [
        "🔍 Google 뉴스 검색",
        "📰 네이버 뉴스 검색",
        "📡 RSS 피드",
        "🔗 URL 직접",
        "✍️ 텍스트 입력",
    ], key="src_radio", label_visibility="collapsed")

    # ── Google 뉴스 ──────────────────────────────────────────
    if src == "🔍 Google 뉴스 검색":
        gkw = st.text_input("검색어", placeholder="교육과정 AI 활용",
                            key="gkw")
        g_max = st.slider("최대 기사", 3, 10, 5, key="g_max")
        if st.button("🔍 Google 뉴스 수집", use_container_width=True):
            if gkw:
                with st.spinner(f"Google 뉴스 '{gkw}' 검색 중..."):
                    try:
                        rss = RSSCollector()
                        docs = rss.search_google_news(gkw, g_max)
                        ok = [d for d in docs if d.status=="ok"]
                        for d in ok:
                            col.collected_docs.append(d)
                            col.total_tokens += d.tokens
                        st.success(f"✓ {len(ok)}개 수집 ({sum(d.tokens for d in ok)}토큰)")
                        for d in ok[:3]:
                            st.caption(f"  • {d.title[:45]}")
                    except Exception as e:
                        st.error(f"수집 실패: {e}")
            else:
                st.warning("검색어를 입력하세요.")

    # ── 네이버 뉴스 ──────────────────────────────────────────
    elif src == "📰 네이버 뉴스 검색":
        nkw = st.text_input("검색어", placeholder="교육 AI",
                            key="nkw")
        with st.expander("네이버 API 설정 (선택)", expanded=False):
            st.session_state.naver_id     = st.text_input(
                "Client ID", value=st.session_state.naver_id, key="nav_id")
            st.session_state.naver_secret = st.text_input(
                "Client Secret",
                value=st.session_state.naver_secret,
                type="password", key="nav_sec")
            st.caption("없으면 네이버 뉴스 섹션 RSS 사용")
        n_max = st.slider("최대 기사", 3, 10, 5, key="n_max")
        if st.button("📰 네이버 뉴스 수집", use_container_width=True):
            if nkw:
                with st.spinner(f"네이버 '{nkw}' 검색 중..."):
                    try:
                        rss = RSSCollector()
                        docs = rss.search_naver_news(
                            nkw,
                            client_id=st.session_state.naver_id,
                            client_secret=st.session_state.naver_secret,
                            max_items=n_max,
                        )
                        ok = [d for d in docs if d.status=="ok"]
                        for d in ok:
                            col.collected_docs.append(d)
                            col.total_tokens += d.tokens
                        st.success(f"✓ {len(ok)}개 수집")
                        for d in ok[:3]:
                            st.caption(f"  • {d.title[:45]}")
                    except Exception as e:
                        st.error(f"수집 실패: {e}")
            else:
                st.warning("검색어를 입력하세요.")

    # ── RSS 피드 ─────────────────────────────────────────────
    elif src == "📡 RSS 피드":
        rss_presets = {
            "직접 입력": "",
            "Google 뉴스 교육":
                "https://news.google.com/rss/search?q=교육&hl=ko&gl=KR&ceid=KR:ko",
            "Google 뉴스 AI":
                "https://news.google.com/rss/search?q=인공지능&hl=ko&gl=KR&ceid=KR:ko",
            "네이버 교육":
                "https://news.naver.com/rss/section_083.xml",
            "네이버 IT과학":
                "https://news.naver.com/rss/section_105.xml",
            "연합뉴스 교육":
                "https://www.yonhapnews.co.kr/RSS/education.xml",
            "위키백과 최근변경":
                "https://ko.wikipedia.org/w/api.php?action=feedrecentchanges&lang=ko&feedformat=rss",
        }
        preset  = st.selectbox("프리셋", list(rss_presets.keys()),
                               key="rss_preset")
        rss_url = st.text_input("RSS URL",
                                value=rss_presets[preset], key="rss_url")
        r_max   = st.slider("최대 기사", 3, 15, 5, key="r_max")
        if st.button("📡 RSS 수집", use_container_width=True):
            if rss_url:
                with st.spinner("RSS 수집 중..."):
                    try:
                        rss_col = RSSCollector()
                        docs = rss_col.collect_feed(rss_url, r_max)
                        ok = [d for d in docs if d.status=="ok"]
                        for d in ok:
                            col.collected_docs.append(d)
                            col.total_tokens += d.tokens
                        st.success(f"✓ {len(ok)}/{len(docs)}개 수집")
                    except Exception as e:
                        st.error(f"RSS 실패: {e}")
            else:
                st.warning("URL을 입력하세요.")

    # ── URL 직접 ─────────────────────────────────────────────
    elif src == "🔗 URL 직접":
        urls_input = st.text_area("URL (줄마다 하나)",
                                  height=80, key="urls_direct",
                                  placeholder="https://example.com")
        if st.button("📥 수집", use_container_width=True):
            urls = [u.strip() for u in urls_input.split("\n")
                    if u.strip().startswith("http")]
            if urls:
                prog = st.progress(0)
                for i, url in enumerate(urls):
                    doc = col.collect_url(url)
                    prog.progress((i+1)/len(urls))
                    icon = "✓" if doc.status=="ok" else "✗"
                    st.caption(f"{icon} {url[:40]} ({doc.tokens}토큰)")
            else:
                st.warning("URL을 입력하세요.")

    # ── 텍스트 직접 ──────────────────────────────────────────
    elif src == "✍️ 텍스트 입력":
        direct = st.text_area("텍스트", height=100, key="direct_txt",
                              placeholder="직접 입력할 텍스트...")
        if st.button("➕ 추가", use_container_width=True):
            if direct.strip():
                doc = col.collect_text(direct, "직접입력")
                st.success(f"✓ {doc.tokens}토큰 추가")

    # ── 수집 현황 + 학습 ─────────────────────────────────────
    st.markdown("---")
    s_col = col.summary()
    c1,c2,c3 = st.columns(3)
    c1.metric("수집 문서", s_col["성공"])
    c2.metric("총 토큰",   s_col["총 토큰"])
    c3.metric("총 문장",   s_col["총 문장"])

    if col.collected_docs:
        with st.expander(f"수집 목록 ({len(col.collected_docs)}개)"):
            for d in col.collected_docs:
                icon = "✅" if d.status=="ok" else "❌"
                st.caption(f"{icon} {d.title[:40]} ({d.tokens}토큰)")

    ep_col = st.select_slider("학습 강도", [5,10,15,20], value=10,
                              key="ep_col")

    if st.button("🚀 수집 내용으로 학습",
                 use_container_width=True,
                 disabled=s_col["성공"]==0):
        corpus = col.get_merged_corpus()
        if corpus.strip():
            prog = st.progress(0)
            def cb2(pct,msg): prog.progress(pct)
            with st.spinner("수집 코퍼스 학습 중..."):
                core.build(corpus, epochs=ep_col, on_progress=cb2)
                st.session_state.ready = True
            st.success(f"✓ 완료! 어휘 {len(core.polar.word_vecs)}개")
            st.rerun()

    if st.button("🗑️ 수집 초기화", use_container_width=True,
                 key="col_clear"):
        col.clear(); st.rerun()

# ━━━ 섹션 6: 저장 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
with st.expander("💾 저장 / 내보내기", expanded=False):
    st.caption(
        "📱 모바일: 아래 버튼 → 파일 저장 → 다음 접속 시 위에서 불러오기")

    try:
        raw = core.to_bytes()
        size_kb = len(raw)/1024
        st.download_button(
            label=f"💾 AI 전체 저장 ({size_kb:.0f}KB)",
            data=raw,
            file_name=f"polar_ai_{len(core.polar.word_vecs)}w.pkl",
            mime="application/octet-stream",
            use_container_width=True,
            help="극성 임베딩 + 가드레일 + 대화 이력 전부 저장"
        )
    except Exception as e:
        st.error(f"저장 준비 실패: {e}")

    # 대화만 텍스트로 내보내기
    sess = core.memory.get_current()
    if sess and sess.turns:
        lines = []
        for t in sess.turns:
            lines.append(f"[{t.role}] {t.content}")
        st.download_button(
            "📄 대화 텍스트 저장",
            data="\n".join(lines).encode("utf-8"),
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ━━━ 섹션 6: 분석 패널 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("🔬 분석 패널", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**레이어 통계**")
        st.caption(f"SSM 성공: {core.stats.get('ssm_pass',0)}")
        st.caption(f"폴백:     {core.stats.get('fallback',0)}")
        st.caption(f"없음:     {core.stats.get('fatal',0)}")

    with c2:
        st.markdown("**관심 키워드**")
        kws = core.memory.context.get_context_keywords(6)
        if kws:
            for kw in kws[:6]:
                st.caption(f"• {kw}")
        else:
            st.caption("(대화 후 표시)")

    # 직접 XAI 분석
    st.markdown("---")
    st.markdown("**텍스트 직접 분석**")
    xai_q = st.text_input("분석할 텍스트", key="xai_direct",
                           label_visibility="collapsed",
                           placeholder="판정할 텍스트 입력...")
    if xai_q:
        r = core.guard.evaluate(xai_q, thr=st.session_state.logp_thr)
        v = r["status"]
        col = {"PASS":"#00ff88","WARNING":"#ffd600","FATAL":"#ff4444"}[v]
        st.markdown(
            f'<div style="color:{col};font-family:monospace;'
            f'font-size:1.2rem;font-weight:700;">{v}</div>'
            f'<div style="font-size:0.75rem;color:#64748b;">'
            f'logP:{r["logp"]:+.2f} | z:{r["z"]:+.2f}</div>',
            unsafe_allow_html=True)
        # 토큰별
        if r.get("tokens"):
            html = ""
            for tok, lp in r["tokens"]:
                cls = "tok-p" if lp>=-10 else "tok-w" if lp>=-14 else "tok-f"
                html += f'<span class="{cls}">{tok}</span> '
            st.markdown(html, unsafe_allow_html=True)
            st.caption("🟢 정상 | 🟡 경계 | 🔴 이탈")


# 푸터
st.markdown(
    '<div style="text-align:center;color:#1e3050;font-size:0.65rem;'
    'padding:0.5rem;font-family:monospace;">'
    'Polar-SSM · GPU 0 · CPU only · 오프라인 완전 독립'
    '</div>',
    unsafe_allow_html=True)
