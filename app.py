# ============================
# app.py  (COPY + PASTE READY)
# ============================
# What this backend does:
# 1) Takes the visitor's job title + their company website URL
# 2) Scrapes their company website to understand what they do
# 3) Scrapes SGI Europe for recent / relevant SGI links
# 4) Uses AI (if OPENAI_API_KEY is set + has quota) to:
#    - summarize SGI articles
#    - score relevance to the visitor (job + company site)
#    - explain WHY each SGI item is relevant
# 5) If AI is not available (quota/missing key), it falls back to keyword matching.

import os
import re
import json
import asyncio
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI is optional at runtime: the server still works without it (fallback mode)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------
# Config
# -------------------------
SGI_BASE = "https://www.sgieurope.com"
SGI_SEED_PAGES = [
    "https://www.sgieurope.com/",               # homepage
    "https://www.sgieurope.com/home/topics",    # topics hub
]

MAX_SGI_LINKS_COLLECT = 80     # collect up to this many candidate SGI links
MAX_SGI_ARTICLES_FETCH = 18    # fetch up to this many pages for analysis
RETURN_TOP = 6                 # return top N recommendations
TIMEOUT = 15

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AI_ENABLED = bool(OPENAI_API_KEY) and (OpenAI is not None)

client = OpenAI(api_key=OPENAI_API_KEY) if AI_ENABLED else None


# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()

# Allow widget usage from GitHub Pages + SGI + anywhere during testing
# (You can tighten this later.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to ["https://www.sgieurope.com", "https://qfngb8kfc6-bot.github.io"] when ready
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Models
# -------------------------
class RecommendRequest(BaseModel):
    job_title: str
    website_url: str


# -------------------------
# Health
# -------------------------
@app.get("/")
def root():
    return {
        "status": "SGI Recommender API is running",
        "ai_enabled": AI_ENABLED
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# Utils: HTML -> clean text
# -------------------------
def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

def same_domain(url: str, base: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(base).netloc
    except Exception:
        return False

def soup_clean_text(html: str, max_chars: int = 7000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Pull extra context from meta
    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = md.get("content").strip()

    # headings + paragraphs are usually best signal
    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if meta_desc:
        parts.append(f"DESCRIPTION: {meta_desc}")

    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t and len(t) > 20:
            parts.append(f"HEADING: {t}")

    para_count = 0
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 60:
            parts.append(t)
            para_count += 1
        if para_count >= 12:
            break

    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# -------------------------
# Fetchers
# -------------------------
async def fetch_html(http: httpx.AsyncClient, url: str) -> str:
    r = await http.get(url, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return r.text

async def fetch_text(http: httpx.AsyncClient, url: str, max_chars: int = 7000) -> str:
    html = await fetch_html(http, url)
    return soup_clean_text(html, max_chars=max_chars)


# -------------------------
# SGI link discovery
# -------------------------
SGI_URL_ALLOW = re.compile(
    r"(\/\d{6}\.article$)|"      # .../119694.article
    r"(\/home\/topics)|"         # topics hubs
    r"(\/brands\/)|"
    r"(\/retail\/)|"
    r"(\/financial\/)|"
    r"(\/technology\/)|"
    r"(\/legal\/)|"
    r"(\/distribution\/)|"
    r"(\/sourcing\/)|"
    r"(\/consumer\/)|"
    r"(\/corporate\/)|"
    r"(\/people\/)|"
    r"(\/market)|"
    r"(\/advisory)|"
    r"(\/reports)"
)

def normalize_url(url: str) -> str:
    # remove fragments
    url = url.split("#")[0].strip()
    return url

async def collect_sgi_links(http: httpx.AsyncClient) -> List[str]:
    links = []
    seen = set()

    for seed in SGI_SEED_PAGES:
        try:
            html = await fetch_html(http, seed)
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href:
                continue
            full = normalize_url(urljoin(SGI_BASE, href))
            if not same_domain(full, SGI_BASE):
                continue
            if full in seen:
                continue
            # basic SGI relevance filtering
            if SGI_URL_ALLOW.search(full):
                seen.add(full)
                links.append(full)
            if len(links) >= MAX_SGI_LINKS_COLLECT:
                break

        if len(links) >= MAX_SGI_LINKS_COLLECT:
            break

    # Prefer actual articles
    articles = [u for u in links if u.endswith(".article")]
    others = [u for u in links if not u.endswith(".article")]

    # return with articles first
    return (articles + others)[:MAX_SGI_LINKS_COLLECT]


# -------------------------
# Heuristic fallback relevance (no AI)
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

def heuristic_score(job: str, company_text: str, article_text: str) -> Dict[str, Any]:
    base = " ".join([job or "", company_text or ""])
    k1 = set(keywords(base, 25))
    k2 = set(keywords(article_text, 25))
    overlap = k1.intersection(k2)
    score = min(100, int(len(overlap) * 10))
    reason = "Matched keywords: " + (", ".join(sorted(list(overlap))[:10]) if overlap else "no strong keyword overlap")
    return {"score": score, "reason": reason}


# -------------------------
# AI helpers
# -------------------------
async def ai_summarize(text: str) -> str:
    if not AI_ENABLED:
        # quick fallback summary
        return (text[:260] + "…") if len(text) > 260 else text

    prompt = (
        "Summarize the SGI page below in 2-3 concise sentences for a busy business professional. "
        "Focus on what happened and why it matters.\n\n"
        f"CONTENT:\n{text[:5000]}"
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
    except Exception:
        # quota / network / etc
        return (text[:260] + "…") if len(text) > 260 else text

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

        # Attempt strict JSON parse
        data = json.loads(content)
        score = int(max(0, min(100, float(data.get("score", 0)))))
        reason = str(data.get("reason", "")).strip()[:400]
        if not reason:
            reason = "Relevant based on job role and company website signals."
        return {"score": score, "reason": reason}

    except Exception:
        hs = heuristic_score(job, company_text, sgi_summary)
        return {"score": hs["score"], "reason": hs["reason"]}


# -------------------------
# Parse SGI article page
# -------------------------
def extract_title_from_html(html: str) -> str:
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

async def fetch_and_build_candidate(http: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
    try:
        html = await fetch_html(http, url)
    except Exception:
        return None

    title = extract_title_from_html(html)
    text = soup_clean_text(html, max_chars=8500)

    # If page is very thin, ignore
    if len(text) < 300:
        return None

    summary = await ai_summarize(text)

    return {
        "title": title,
        "url": url,
        "raw_text": text,
        "summary": summary,
    }


# -------------------------
# Main recommendation flow
# -------------------------
async def recommend_for_user(job_title: str, website_url: str) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    website_url = ensure_http(website_url)

    if not job_title or len(job_title) < 2:
        raise HTTPException(status_code=422, detail="job_title is required.")
    if not website_url:
        raise HTTPException(status_code=422, detail="website_url is required.")

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as http:
        # 1) Scrape user's website
        try:
            company_text = await fetch_text(http, website_url, max_chars=6500)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch website_url: {str(e)}")

        # 2) Collect SGI links
        sgi_links = await collect_sgi_links(http)
        if not sgi_links:
            return {"articles": [], "error": None}

        # 3) Fetch SGI pages concurrently (limit)
        selected = sgi_links[:MAX_SGI_ARTICLES_FETCH]
        tasks = [fetch_and_build_candidate(http, u) for u in selected]
        candidates = await asyncio.gather(*tasks)
        candidates = [c for c in candidates if c]

        if not candidates:
            return {"articles": [], "error": None}

        # 4) Score relevance (AI or heuristic)
        scored = []
        for c in candidates:
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

        # 5) Return top
        top = scored[:RETURN_TOP]
        return {"articles": top, "error": None}


# -------------------------
# API Endpoint
# -------------------------
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        # IMPORTANT: This endpoint accepts job_title + website_url (not "url" / "client_niche")
        return await recommend_for_user(req.job_title, req.website_url)

    except HTTPException:
        raise

    except Exception as e:
        # Always log the real error for Render logs
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
