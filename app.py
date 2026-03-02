# ============================================================
# app.py  (COPY + PASTE READY)
# SGI-ONLY RECOMMENDER + REAL PROGRESS (SSE)
# ============================================================
# ✅ Hard guarantee: ONLY returns links from https://www.sgieurope.com
#
# What you get:
# 1) POST /recommend        -> (blocking) returns {articles:[...], error:null}
# 2) POST /recommend_async  -> returns {task_id}
# 3) GET  /progress/{id}    -> Server-Sent Events stream:
#        event: progress
#        data: {"pct": 42, "step": "...", "done": false}
#      and final:
#        event: done
#        data: {"pct":100,"step":"Done","done":true,"result":{...}}
#
# Your widget progress bar becomes REAL by using /recommend_async + /progress/{id}.
# ============================================================

import os
import re
import json
import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from starlette.responses import StreamingResponse

# OpenAI optional: app runs without it (keyword fallback)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------
# Config
# -------------------------
SGI_BASE = "https://www.sgieurope.com"
SGI_HOST = urlparse(SGI_BASE).netloc

SEED_PAGES = [
    "https://www.sgieurope.com/",
    "https://www.sgieurope.com/home/topics",
]

TIMEOUT = 15
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Crawl limits (keep sane for Render free tier)
MAX_TOPIC_PAGES = 14
MAX_LINKS_TOTAL = 220
MAX_PAGES_FETCH = 26
RETURN_TOP = 6

# AI optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AI_ENABLED = bool(OPENAI_API_KEY) and (OpenAI is not None)
client = OpenAI(api_key=OPENAI_API_KEY) if AI_ENABLED else None


# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()

# Allow widget usage from anywhere while testing.
# Tighten later to ["https://www.sgieurope.com"] if embedding on SGI only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# In-memory progress store
# -------------------------
# NOTE: In-memory works fine for a single Render instance.
# If you scale to multiple instances, move this to Redis.
TASKS: Dict[str, Dict[str, Any]] = {}
TASK_LOCK = asyncio.Lock()
TASK_TTL_SECONDS = 60 * 15  # 15 minutes


async def set_task(task_id: str, **updates):
    async with TASK_LOCK:
        t = TASKS.get(task_id)
        if not t:
            return
        t.update(updates)
        t["updated_at"] = time.time()


async def create_task_record(task_id: str):
    async with TASK_LOCK:
        TASKS[task_id] = {
            "pct": 0,
            "step": "Starting…",
            "done": False,
            "error": None,
            "result": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }


async def cleanup_tasks():
    """Remove old tasks occasionally."""
    now = time.time()
    async with TASK_LOCK:
        dead = [k for k, v in TASKS.items() if (now - v.get("updated_at", now)) > TASK_TTL_SECONDS]
        for k in dead:
            TASKS.pop(k, None)


# -------------------------
# Models
# -------------------------
class RecommendRequest(BaseModel):
    # Accept BOTH styles:
    # - new: job_title + website_url
    # - old widget: url + job_title
    job_title: Optional[str] = None
    website_url: Optional[str] = None

    # compat fields the widget may send
    url: Optional[str] = None
    profile: Optional[str] = None
    intent: Optional[str] = None

    @model_validator(mode="after")
    def normalize_fields(self):
        # Map url -> website_url if website_url missing
        if not self.website_url and self.url:
            self.website_url = self.url

        # If job_title missing but profile exists, try a very simple extraction
        if not self.job_title and self.profile:
            # naive attempt: take text before " at " if present
            p = self.profile.strip()
            if " at " in p.lower():
                self.job_title = p.split(" at ")[0].strip()
            else:
                # otherwise just use profile as job_title fallback (better than nothing)
                self.job_title = p[:80].strip()

        if not self.job_title or not self.website_url:
            raise ValueError("job_title and website_url are required (or send {url, job_title}).")

        return self


# -------------------------
# Health
# -------------------------
@app.get("/")
def root():
    return {"status": "SGI-ONLY Recommender API running", "ai_enabled": AI_ENABLED}

@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# URL helpers (SGI-only enforcement)
# -------------------------
def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    url = url.split("#")[0]
    if url.endswith("/") and len(url) > len("https://x/"):
        url = url[:-1]
    return url

def is_sgi_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and u.netloc == SGI_HOST
    except Exception:
        return False

def absolutize_sgi(href: str) -> Optional[str]:
    if not href:
        return None
    full = normalize_url(urljoin(SGI_BASE, href))
    return full if is_sgi_url(full) else None


# -------------------------
# HTML -> text helpers
# -------------------------
def soup_clean_text(html: str, max_chars: int = 8000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    parts = []

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        parts.append(f"TITLE: {title}")

    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        parts.append(f"DESCRIPTION: {md.get('content','').strip()}")

    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t and len(t) > 20:
            parts.append(f"HEADING: {t}")

    p_count = 0
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 60:
            parts.append(t)
            p_count += 1
        if p_count >= 14:
            break

    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        if t:
            return t
    if soup.title:
        t = soup.title.get_text(" ", strip=True)
        if t:
            return t
    return "No title"


# -------------------------
# Fetch helpers
# -------------------------
async def fetch_html(http: httpx.AsyncClient, url: str) -> str:
    r = await http.get(url, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return r.text


# -------------------------
# Crawl SGI (SGI-only)
# -------------------------
PREFER_PATTERNS = re.compile(
    r"(\.article$)|"
    r"(/home/topics)|"
    r"(/brands/)|(/retail/)|(/financial/)|(/technology/)|(/legal/)|"
    r"(/distribution/)|(/sourcing/)|(/consumer/)|(/corporate/)|(/people/)|"
    r"(/manufacture/)|(/community/)|(/market)"
)

def sort_prefer(urls: List[str]) -> List[str]:
    preferred = [u for u in urls if PREFER_PATTERNS.search(u.lower())]
    other = [u for u in urls if u not in preferred]
    return preferred + other

async def discover_topic_pages(http: httpx.AsyncClient) -> List[str]:
    topics_url = "https://www.sgieurope.com/home/topics"
    try:
        html = await fetch_html(http, topics_url)
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    topic_pages: List[str] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        full = absolutize_sgi(a.get("href", ""))
        if not full:
            continue
        if "/home/topics/" in full.lower() and full not in seen:
            seen.add(full)
            topic_pages.append(full)
        if len(topic_pages) >= MAX_TOPIC_PAGES:
            break

    return topic_pages

async def collect_sgi_links(http: httpx.AsyncClient) -> List[str]:
    to_crawl = list(SEED_PAGES)
    topic_pages = await discover_topic_pages(http)
    to_crawl.extend(topic_pages)

    seen_pages: Set[str] = set()
    seen_links: Set[str] = set()
    collected: List[str] = []

    for page in to_crawl:
        page = normalize_url(page)
        if page in seen_pages:
            continue
        seen_pages.add(page)

        try:
            html = await fetch_html(http, page)
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            full = absolutize_sgi(a.get("href", ""))
            if not full:
                continue

            lower = full.lower()
            if any(x in lower for x in ["/login", "/subscribe", "/account", "/search?"]):
                continue

            if full in seen_links:
                continue

            seen_links.add(full)
            collected.append(full)

            if len(collected) >= MAX_LINKS_TOTAL:
                break

        if len(collected) >= MAX_LINKS_TOTAL:
            break

    collected = sort_prefer(collected)
    collected = [u for u in collected if is_sgi_url(u)]
    return collected[:MAX_LINKS_TOTAL]


# -------------------------
# Fallback relevance (no AI)
# -------------------------
STOP = set("""
a an the and or for to of in on at with from by as is are be this that these those
your you we our they their it its into about over under between across
""".split())

def keywords(text: str, max_k: int = 30) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    toks = [t for t in text.split() if len(t) >= 4 and t not in STOP]
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:max_k]]

def heuristic_score(job: str, company_text: str, sgi_text: str) -> Dict[str, Any]:
    base = " ".join([job or "", company_text or ""])
    k1 = set(keywords(base, 25))
    k2 = set(keywords(sgi_text, 25))
    overlap = k1.intersection(k2)
    score = min(100, int(len(overlap) * 10))
    reason = "Matched keywords: " + (", ".join(sorted(list(overlap))[:10]) if overlap else "no strong keyword overlap")
    return {"score": score, "reason": reason}


# -------------------------
# AI helpers (optional; graceful fallback on quota)
# -------------------------
def is_openai_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("insufficient_quota" in msg) or ("error code: 429" in msg) or ("quota" in msg)

async def ai_summarize(text: str) -> str:
    if not AI_ENABLED:
        return (text[:280] + "…") if len(text) > 280 else text

    prompt = (
        "Summarize the SGI page content below in 2-3 concise sentences for a busy business professional. "
        "Focus on what happened and why it matters.\n\n"
        f"CONTENT:\n{text[:5500]}"
    )

    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You write crisp, factual business summaries."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # quota / outage -> fallback
        return (text[:280] + "…") if len(text) > 280 else text

async def ai_relevance(job: str, company_text: str, sgi_title: str, sgi_url: str, sgi_summary: str) -> Dict[str, Any]:
    if not AI_ENABLED:
        hs = heuristic_score(job, company_text, sgi_summary)
        return {"score": hs["score"], "reason": hs["reason"]}

    prompt = f"""
You are matching SGI Europe content to a website visitor.

VISITOR JOB TITLE:
{job}

VISITOR COMPANY WEBSITE SIGNAL (scraped):
{company_text[:3500]}

SGI ITEM:
Title: {sgi_title}
URL: {sgi_url}
Summary: {sgi_summary}

Task:
1) Give a relevance score from 0-100
2) Explain WHY it is relevant in 1-2 sentences, specifically referencing the visitor job + what their company seems to do.
Return STRICT JSON only with keys: score (number), reason (string).
"""

    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Return JSON only. No markdown. No extra keys."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)

        score = int(max(0, min(100, float(data.get("score", 0)))))
        reason = str(data.get("reason", "")).strip()[:450]
        if not reason:
            reason = "Relevant based on your role and your company website signals."
        return {"score": score, "reason": reason}

    except Exception:
        hs = heuristic_score(job, company_text, sgi_summary)
        return {"score": hs["score"], "reason": hs["reason"]}


# -------------------------
# Build SGI candidate (SGI-only)
# -------------------------
async def fetch_sgi_candidate(http: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
    if not is_sgi_url(url):
        return None

    try:
        html = await fetch_html(http, url)
    except Exception:
        return None

    title = extract_title(html)
    text = soup_clean_text(html, max_chars=9000)

    if len(text) < 350:
        return None

    summary = await ai_summarize(text)

    return {
        "title": title,
        "url": url,
        "raw_text": text,
        "summary": summary,
    }


# -------------------------
# Main recommendation flow (WITH progress callbacks)
# -------------------------
async def recommend_for_user(
    job_title: str,
    website_url: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    website_url = ensure_http(website_url)

    if not job_title:
        raise HTTPException(status_code=422, detail="job_title is required.")
    if not website_url:
        raise HTTPException(status_code=422, detail="website_url is required.")

    async def bump(pct: int, step: str):
        if task_id:
            await set_task(task_id, pct=pct, step=step)

    await bump(5, "Validating inputs…")

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as http:
        # 1) Scrape visitor company website
        await bump(12, "Fetching your website…")
        try:
            company_html = await fetch_html(http, website_url)
            company_text = soup_clean_text(company_html, max_chars=6500)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch website_url: {str(e)}")

        await bump(28, "Understanding what your company does…")

        # 2) Collect SGI-only links
        await bump(40, "Collecting SGI Europe topics & links…")
        sgi_links = await collect_sgi_links(http)
        sgi_links = [u for u in sgi_links if is_sgi_url(u)]

        if not sgi_links:
            await bump(100, "Done")
            return {"articles": [], "error": None}

        # 3) Fetch SGI pages concurrently (limit)
        await bump(55, "Fetching SGI pages…")
        selected = sgi_links[:MAX_PAGES_FETCH]
        candidates = await asyncio.gather(*[fetch_sgi_candidate(http, u) for u in selected])
        candidates = [c for c in candidates if c and is_sgi_url(c["url"])]

        if not candidates:
            await bump(100, "Done")
            return {"articles": [], "error": None}

        # 4) Score relevance
        await bump(72, "Scoring relevance to your role…")
        scored = []
        for i, c in enumerate(candidates):
            # small progress increments during scoring
            step_pct = 72 + int((i / max(1, len(candidates))) * 18)
            await bump(step_pct, f"Ranking SGI pages… ({i+1}/{len(candidates)})")

            rel = await ai_relevance(
                job=job_title,
                company_text=company_text,
                sgi_title=c["title"],
                sgi_url=c["url"],
                sgi_summary=c["summary"],
            )
            scored.append({
                "title": c["title"],
                "url": c["url"],
                "summary": c["summary"],
                "relevance_score": rel["score"],
                "relevance_reason": rel["reason"],
            })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        await bump(94, "Finalising top results…")

        # 5) Return top (SGI-only guarantee)
        top = [a for a in scored if is_sgi_url(a["url"])][:RETURN_TOP]

        await bump(100, "Done")
        return {"articles": top, "error": None}


# -------------------------
# Blocking endpoint (works with simple widget fetch)
# -------------------------
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        return await recommend_for_user(req.job_title, req.website_url, task_id=None)
    except HTTPException:
        raise
    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Async job start (for REAL progress bar)
# -------------------------
@app.post("/recommend_async")
async def recommend_async(req: RecommendRequest):
    """
    Returns a task_id immediately, then compute runs in background.
    Widget connects to /progress/{task_id} to receive progress + final result.
    """
    await cleanup_tasks()

    task_id = uuid.uuid4().hex
    await create_task_record(task_id)
    await set_task(task_id, pct=1, step="Queued…")

    async def runner():
        try:
            result = await recommend_for_user(req.job_title, req.website_url, task_id=task_id)
            await set_task(task_id, done=True, pct=100, step="Done", result=result, error=None)
        except HTTPException as he:
            await set_task(task_id, done=True, pct=100, step="Error", result=None, error=str(he.detail))
        except Exception as e:
            await set_task(task_id, done=True, pct=100, step="Error", result=None, error=str(e))

    # Fire-and-forget (single instance)
    asyncio.create_task(runner())

    return {"task_id": task_id}


# -------------------------
# SSE progress stream
# -------------------------
@app.get("/progress/{task_id}")
async def progress(task_id: str):
    """
    Server-Sent Events:
    - event: progress  data: {"pct":..,"step":..,"done":false}
    - event: done      data: {"pct":100,"step":"Done","done":true,"result":{...}}
    - event: error     data: {"pct":100,"step":"Error","done":true,"error":"..."}
    """
    async def event_gen():
        # initial check
        async with TASK_LOCK:
            if task_id not in TASKS:
                payload = {"pct": 100, "step": "Unknown task_id", "done": True, "error": "Unknown task_id"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return

        last_sent = None

        while True:
            await asyncio.sleep(0.25)

            async with TASK_LOCK:
                t = TASKS.get(task_id)

            if not t:
                payload = {"pct": 100, "step": "Task expired", "done": True, "error": "Task expired"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return

            snapshot = {
                "pct": int(t.get("pct", 0)),
                "step": t.get("step", ""),
                "done": bool(t.get("done", False)),
            }

            if t.get("error"):
                snapshot["error"] = t["error"]

            if t.get("done") and t.get("result"):
                snapshot["result"] = t["result"]

            # avoid sending identical frames repeatedly
            key = json.dumps(snapshot, sort_keys=True)
            if key != last_sent:
                last_sent = key
                if snapshot.get("done") and snapshot.get("error"):
                    yield f"event: error\ndata: {json.dumps(snapshot)}\n\n"
                    return
                if snapshot.get("done"):
                    yield f"event: done\ndata: {json.dumps(snapshot)}\n\n"
                    return
                else:
                    yield f"event: progress\ndata: {json.dumps(snapshot)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
