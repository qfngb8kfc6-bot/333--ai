# ============================================================
# app.py  (COPY + PASTE READY)
# SGI-ONLY RECOMMENDER (NEVER leaves sgieurope.com)
# ============================================================
# What this backend does:
# 1) Takes: job_title + website_url (visitor company site)
# 2) Scrapes the visitor company site for signals about what they do
# 3) Collects ONLY SGI Europe links (sgieurope.com) from:
#    - Homepage
#    - /home/topics
#    - Each topic page discovered from /home/topics
#    - Plus any additional internal SGI pages linked from those hubs
# 4) Fetches a batch of SGI pages and produces:
#    - short summary (AI if available; fallback otherwise)
#    - relevance score + “why relevant” (AI if available; fallback otherwise)
# 5) Returns top results
#
# ✅ Hard guarantee: links returned are ONLY from https://www.sgieurope.com

import os
import re
import json
import asyncio
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Crawl limits (keep these sane for Render free tier)
MAX_TOPIC_PAGES = 14           # how many topic pages to crawl from /home/topics
MAX_LINKS_TOTAL = 220          # how many SGI URLs to consider (max)
MAX_PAGES_FETCH = 26           # how many SGI pages to fetch & analyze
RETURN_TOP = 6                 # how many results to return

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
    url = url.split("#")[0]  # remove fragments
    # strip trailing slash except root
    if url.endswith("/") and len(url) > len("https://x/"):
        url = url[:-1]
    return url

def is_sgi_url(url: str) -> bool:
    """Hard filter: ONLY SGI host allowed."""
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and u.netloc == SGI_HOST
    except Exception:
        return False

def absolutize_sgi(href: str) -> Optional[str]:
    """Turn href into absolute SGI URL and enforce SGI-only."""
    if not href:
        return None
    full = normalize_url(urljoin(SGI_BASE, href))
    if is_sgi_url(full):
        return full
    return None


# -------------------------
# HTML -> text helpers
# -------------------------
def soup_clean_text(html: str, max_chars: int = 8000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Add meta title/description for signal
    parts = []

    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    if title:
        parts.append(f"TITLE: {title}")

    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        parts.append(f"DESCRIPTION: {md.get('content','').strip()}")

    # Headings for structure
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t and len(t) > 20:
            parts.append(f"HEADING: {t}")

    # A few substantial paragraphs
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
    if not is_sgi_url(url) and urlparse(url).netloc != urlparse(ensure_http(url)).netloc:
        # This guard is mostly for SGI fetches; company URL may be external.
        pass

    r = await http.get(url, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return r.text

async def fetch_text(http: httpx.AsyncClient, url: str, max_chars: int = 8000) -> str:
    html = await fetch_html(http, url)
    return soup_clean_text(html, max_chars=max_chars)


# -------------------------
# Crawl SGI (SGI-only)
# -------------------------
# Accept SGI content patterns, but do NOT rely solely on them (site structure may change).
# This is just a “priority” hint, not a security filter.
PREFER_PATTERNS = re.compile(
    r"(\.article$)|"            # common SGI article ending
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
    """
    Pull topic pages from /home/topics like:
    /home/topics/retail, /home/topics/financial, etc.
    """
    topics_url = "https://www.sgieurope.com/home/topics"
    try:
        html = await fetch_html(http, topics_url)
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    topic_pages: List[str] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        full = absolutize_sgi(href)
        if not full:
            continue
        # specifically capture /home/topics/... pages
        if "/home/topics/" in full.lower() and full not in seen:
            seen.add(full)
            topic_pages.append(full)
        if len(topic_pages) >= MAX_TOPIC_PAGES:
            break

    return topic_pages

async def collect_sgi_links(http: httpx.AsyncClient) -> List[str]:
    """
    SGI-only link collection:
    - Crawl SEED_PAGES
    - Crawl discovered topic pages from /home/topics
    - Collect internal links from those pages
    - Return a prioritized list of SGI URLs
    """
    to_crawl = list(SEED_PAGES)

    # Add topic pages
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

            # hard SGI-only guarantee:
            if not is_sgi_url(full):
                continue

            # avoid obvious junk routes
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

    # prioritize likely article/content pages first
    collected = sort_prefer(collected)

    # also ensure SGI-only (again, belt + braces)
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
# AI helpers (optional)
# -------------------------
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
    except Exception:
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
    # Hard stop: never fetch non-SGI
    if not is_sgi_url(url):
        return None

    try:
        html = await fetch_html(http, url)
    except Exception:
        return None

    title = extract_title(html)
    text = soup_clean_text(html, max_chars=9000)

    # Filter out ultra-thin pages
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
# Main recommendation flow
# -------------------------
async def recommend_for_user(job_title: str, website_url: str) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    website_url = ensure_http(website_url)

    if not job_title:
        raise HTTPException(status_code=422, detail="job_title is required.")
    if not website_url:
        raise HTTPException(status_code=422, detail="website_url is required.")

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as http:
        # 1) Scrape visitor company website (can be any domain)
        try:
            company_html = await fetch_html(http, website_url)
            company_text = soup_clean_text(company_html, max_chars=6500)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch website_url: {str(e)}")

        # 2) Collect SGI-only links
        sgi_links = await collect_sgi_links(http)
        sgi_links = [u for u in sgi_links if is_sgi_url(u)]  # hard re-check

        if not sgi_links:
            return {"articles": [], "error": None}

        # 3) Fetch SGI pages concurrently (limit)
        selected = sgi_links[:MAX_PAGES_FETCH]
        candidates = await asyncio.gather(*[fetch_sgi_candidate(http, u) for u in selected])
        candidates = [c for c in candidates if c and is_sgi_url(c["url"])]

        if not candidates:
            return {"articles": [], "error": None}

        # 4) Score relevance
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

        # 5) Return top (SGI-only guarantee)
        top = [a for a in scored if is_sgi_url(a["url"])][:RETURN_TOP]
        return {"articles": top, "error": None}


# -------------------------
# API endpoint
# -------------------------
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        return await recommend_for_user(req.job_title, req.website_url)
    except HTTPException:
        raise
    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
