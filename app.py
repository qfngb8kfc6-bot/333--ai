# ============================================================
# app.py  (COPY + PASTE READY)
# SGI-ONLY RECOMMENDER (RSS + CACHE + FAST)
# Uses: https://www.sgieurope.com/3764.fullrss
# ============================================================

import os
import re
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI optional (app runs without it)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------
# Config
# -------------------------
SGI_BASE = "https://www.sgieurope.com"
SGI_HOST = urlparse(SGI_BASE).netloc

SGI_RSS_URL = "https://www.sgieurope.com/3764.fullrss"

TIMEOUT = 12
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Speed knobs
RSS_CACHE_SECONDS = 20 * 60           # refresh RSS every 20 minutes
RSS_MAX_ITEMS = 40                   # how many RSS items to keep in cache
CANDIDATES_FOR_AI = 10               # how many to run AI on (keep small!)
RETURN_TOP = 6                       # return top N

# If you want even faster, set this True to avoid fetching full article HTML
USE_RSS_DESCRIPTION_ONLY = True

# Concurrency limits
FETCH_CONCURRENCY = 8

# AI optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AI_ENABLED = bool(OPENAI_API_KEY) and (OpenAI is not None)
client = OpenAI(api_key=OPENAI_API_KEY) if AI_ENABLED else None


# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()

# allow widget usage anywhere while testing
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
    return {
        "status": "SGI RSS Recommender running",
        "ai_enabled": AI_ENABLED,
        "rss": SGI_RSS_URL,
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# Helpers
# -------------------------
def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
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

def soup_clean_text(html: str, max_chars: int = 6500) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    parts = []

    if soup.title:
        t = soup.title.get_text(" ", strip=True)
        if t:
            parts.append(f"TITLE: {t}")

    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        parts.append(f"DESCRIPTION: {md.get('content','').strip()}")

    # Grab a few meaningful headings
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t and len(t) > 12:
            parts.append(f"HEADING: {t}")
        if len(parts) >= 10:
            break

    # Grab substantial paragraphs
    p_count = 0
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 60:
            parts.append(t)
            p_count += 1
        if p_count >= 12:
            break

    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text[:max_chars]


def quick_company_signal(html: str) -> str:
    """
    Very fast extraction from company homepage:
    - title/meta description
    - a few headings/paragraphs
    """
    return soup_clean_text(html, max_chars=4500)


# -------------------------
# RSS Cache (in-memory)
# -------------------------
_RSS_CACHE: Dict[str, Any] = {
    "ts": 0,
    "items": [],   # list of {title, link, description, pubDate}
}

def _parse_rss_xml(xml_text: str) -> List[Dict[str, str]]:
    """
    Parse RSS XML without extra dependencies.
    Returns list of items with title/link/description/pubDate.
    """
    import xml.etree.ElementTree as ET

    # Some RSS feeds have namespaces; handle loosely
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()

        link = normalize_url(link)

        # SGI-only hard filter
        if not link or not is_sgi_url(link):
            continue

        items.append({
            "title": title or "Untitled",
            "url": link,
            "description": desc,
            "pubDate": pub,
        })

    return items


async def _refresh_rss_if_needed(http: httpx.AsyncClient) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Returns (items, from_cache)
    """
    now = time.time()
    if _RSS_CACHE["items"] and (now - _RSS_CACHE["ts"] < RSS_CACHE_SECONDS):
        return _RSS_CACHE["items"], True

    r = await http.get(SGI_RSS_URL, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()

    items = _parse_rss_xml(r.text)
    items = items[:RSS_MAX_ITEMS]

    _RSS_CACHE["ts"] = now
    _RSS_CACHE["items"] = items
    return items, False


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

def heuristic_score(job_title: str, company_signal: str, sgi_title: str, sgi_desc: str) -> Tuple[int, str]:
    base = " ".join([job_title or "", company_signal or ""])
    k1 = set(keywords(base, 28))
    k2 = set(keywords(" ".join([sgi_title or "", sgi_desc or ""]), 28))
    overlap = k1.intersection(k2)
    score = min(100, int(len(overlap) * 9))
    if score < 20 and (job_title or "").strip():
        # small floor so you still get something
        score = min(35, score + 12)

    reason = "Matched keywords: " + (", ".join(sorted(list(overlap))[:10]) if overlap else "no strong overlap")
    return score, reason


# -------------------------
# AI scoring + summary (single call per item)
# -------------------------
async def ai_score_and_summarize(
    job_title: str,
    company_signal: str,
    sgi_title: str,
    sgi_url: str,
    sgi_text: str,
) -> Dict[str, Any]:
    """
    One AI call that returns:
    - summary (2-3 sentences)
    - score 0-100
    - reason (1-2 sentences, personalized)
    """
    if not AI_ENABLED:
        # fallback
        score, reason = heuristic_score(job_title, company_signal, sgi_title, sgi_text)
        # "summary" fallback: truncate
        clean = re.sub(r"\s+", " ", (sgi_text or "")).strip()
        summary = clean[:260] + ("…" if len(clean) > 260 else "")
        return {"score": score, "reason": reason, "summary": summary}

    prompt = f"""
You match SGI Europe content to a website visitor.

VISITOR JOB TITLE:
{job_title}

VISITOR COMPANY WEBSITE SIGNAL (scraped):
{company_signal[:3000]}

SGI ITEM:
Title: {sgi_title}
URL: {sgi_url}

CONTENT / SNIPPET:
{sgi_text[:3500]}

Return STRICT JSON only with these keys:
{{
  "summary": "2-3 concise sentences. No fluff.",
  "score": 0-100,
  "reason": "1-2 sentences explaining why this matters to THIS visitor role and company."
}}
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

        summary = str(data.get("summary", "")).strip()
        reason = str(data.get("reason", "")).strip()
        score_raw = data.get("score", 0)

        try:
            score = int(max(0, min(100, float(score_raw))))
        except Exception:
            score = 0

        if not summary:
            summary = (sgi_text[:260] + "…") if len(sgi_text) > 260 else sgi_text
        if not reason:
            reason = "Relevant based on your role and your company website signals."

        return {"score": score, "reason": reason[:500], "summary": summary[:700]}

    except Exception:
        # fallback if quota/network error
        score, reason = heuristic_score(job_title, company_signal, sgi_title, sgi_text)
        clean = re.sub(r"\s+", " ", (sgi_text or "")).strip()
        summary = clean[:260] + ("…" if len(clean) > 260 else "")
        return {"score": score, "reason": reason, "summary": summary}


# -------------------------
# Fetch SGI article text (optional)
# -------------------------
async def fetch_sgi_text(http: httpx.AsyncClient, url: str) -> str:
    if not is_sgi_url(url):
        return ""
    r = await http.get(url, timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return soup_clean_text(r.text, max_chars=6500)


# -------------------------
# Main endpoint logic
# -------------------------
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    job_title = (req.job_title or "").strip()
    website_url = ensure_http(req.website_url)

    if not job_title:
        raise HTTPException(status_code=422, detail="job_title is required.")
    if not website_url:
        raise HTTPException(status_code=422, detail="website_url is required.")

    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as http:
        # 1) Scrape visitor company website (only 1 request)
        try:
            company_resp = await http.get(website_url, timeout=TIMEOUT, follow_redirects=True)
            company_resp.raise_for_status()
            company_signal = quick_company_signal(company_resp.text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch website_url: {str(e)}")

        # 2) Get SGI RSS items (cached)
        try:
            rss_items, from_cache = await _refresh_rss_if_needed(http)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch SGI RSS feed: {str(e)}")

        if not rss_items:
            return {"articles": [], "error": None, "meta": {"source": "rss", "cached": from_cache}}

        # 3) Cheap pre-rank using heuristic overlap on title + rss description
        ranked = []
        for item in rss_items:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            desc = item.get("description", "") or ""
            if not url or not is_sgi_url(url):
                continue
            score, _ = heuristic_score(job_title, company_signal, title, desc)
            ranked.append((score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Keep only top candidates for AI (fast)
        candidates = [it for _, it in ranked[:max(CANDIDATES_FOR_AI, RETURN_TOP)]]

        # 4) Prepare content for each candidate
        async def build_one(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            url = item.get("url", "")
            if not is_sgi_url(url):
                return None

            title = item.get("title", "Untitled")
            desc = re.sub(r"<[^>]+>", " ", (item.get("description") or ""))  # strip any HTML in RSS desc
            desc = re.sub(r"\s+", " ", desc).strip()

            # Option: use RSS description only (fastest)
            if USE_RSS_DESCRIPTION_ONLY and desc:
                sgi_text = desc
            else:
                # fetch full SGI page text (slower)
                async with sem:
                    try:
                        sgi_text = await fetch_sgi_text(http, url)
                    except Exception:
                        sgi_text = desc or ""

            if not sgi_text or len(sgi_text) < 80:
                sgi_text = desc or ""

            ai = await ai_score_and_summarize(
                job_title=job_title,
                company_signal=company_signal,
                sgi_title=title,
                sgi_url=url,
                sgi_text=sgi_text,
            )

            return {
                "title": title,
                "url": url,
                "summary": ai["summary"],
                "relevance_score": ai["score"],
                "relevance_reason": ai["reason"],
            }

        results = await asyncio.gather(*[build_one(it) for it in candidates])
        results = [r for r in results if r and is_sgi_url(r["url"])]

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        top = results[:RETURN_TOP]

        return {
            "articles": top,
            "error": None,
            "meta": {
                "source": "rss",
                "cached": from_cache,
                "ai_enabled": AI_ENABLED,
                "rss_items_seen": len(rss_items),
                "candidates_scored": len(candidates),
                "use_rss_description_only": USE_RSS_DESCRIPTION_ONLY,
            },
        }
