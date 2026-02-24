import asyncio
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================================
# SGI EUROPE: SITE-SPECIFIC CONTENT RECOMMENDER (PRODUCTION)
# Domain: https://www.sgieurope.com/
# Input: visitor "profile" (role + interests)
# Output: top relevant SGI pages (prefer .article) + clean "why relevant"
# ==========================================================

# --------------------------
# CONFIG
# --------------------------
BASE_SITE = "https://www.sgieurope.com"
MAX_LINKS_FROM_HOME = 60          # how many links we collect from homepage
MAX_PAGES_TO_FETCH = 25           # how many pages we fetch/analyze per request
TOP_RESULTS = 5                   # how many results to return
MIN_TEXT_LEN = 300                # skip pages with too little readable content
HUB_SCORE_THRESHOLD = 90          # require very high score to keep hub/category pages
REQUEST_TIMEOUT = 12.0

# --------------------------
# APP SETUP
# --------------------------
app = FastAPI(title="SGI Europe Recommender", version="1.0.0")

# Allow widget to call your API from SGI Europe (or anywhere while testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Later: set to ["https://www.sgieurope.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# REQUEST/RESPONSE MODELS
# --------------------------
class ProfileRequest(BaseModel):
    profile: str


# --------------------------
# TEXT CLEANING / STOPWORDS
# --------------------------
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was",
    "were", "are", "but", "not", "you", "your", "about", "into", "over", "under",
    "their", "there", "they", "them", "what", "when", "where", "which", "while",
    "who", "whom", "why", "how", "will", "would", "can", "could", "should", "may",
    "might", "been", "being", "also", "than", "then", "its", "it's", "our", "out",
    "off", "per", "via", "due", "all", "any", "each", "other", "on", "in", "at",
    "to", "of", "by", "as", "or", "if", "we", "us", "it", "an", "a"
}

def clean_words(text: str) -> List[str]:
    """
    - keeps only alphabetic tokens length >= 4
    - lowercases
    - removes stopwords
    """
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS]


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# --------------------------
# SGI-SPECIFIC URL HELPERS
# --------------------------
def is_sgi_domain(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(BASE_SITE).netloc
    except Exception:
        return False


def is_article_url(url: str) -> bool:
    """
    SGI Europe frequently uses *.article for real articles.
    """
    return url.lower().endswith(".article")


def is_hub_or_index(url: str) -> bool:
    """
    Detect obvious hub/category pages.
    Keep them only if score is extremely high.
    """
    path = (urlparse(url).path or "").lower().strip("/")
    if not path:
        return True

    # Known hub patterns on SGI
    if path.startswith("home/topics"):
        return True
    if path == "home" or path.startswith("home/"):
        return True

    # Very short paths often indicate category hubs (e.g. /about-us is ok, but /brands maybe hub)
    # We'll treat 1-segment paths as more hub-like unless it's a known info page.
    parts = [p for p in path.split("/") if p]
    if len(parts) == 1 and parts[0] not in {"about-us", "contact", "privacy", "terms"}:
        return True

    return False


# --------------------------
# HTML EXTRACTION
# --------------------------
def extract_main_text(soup: BeautifulSoup) -> str:
    # Remove non-content
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Prefer article-like containers if present
    main = soup.find("article") or soup.find("main") or soup.body
    if not main:
        return ""

    # Collect paragraphs (SGI pages usually have content in <p>)
    paragraphs = []
    for p in main.find_all("p"):
        txt = normalize_spaces(p.get_text(" ", strip=True))
        # Keep substantive paragraphs
        if len(txt) >= 60:
            paragraphs.append(txt)

    text = " ".join(paragraphs)
    return normalize_spaces(text)


def extract_title(soup: BeautifulSoup) -> str:
    # Try h1 first
    h1 = soup.find("h1")
    if h1:
        t = normalize_spaces(h1.get_text(" ", strip=True))
        if t:
            return t

    # Fall back to <title>
    if soup.title and soup.title.string:
        t = normalize_spaces(soup.title.string)
        if t:
            return t

    return "Untitled"


# --------------------------
# RELEVANCE SCORING (KEYWORD OVERLAP, CLEAN)
# --------------------------
def score_relevance(page_text: str, profile: str):
    """
    Returns:
      score: 0-100
      matched_words: list of meaningful matched tokens
    """
    profile_words = clean_words(profile)
    page_word_set = set(clean_words(page_text))

    matched = [w for w in profile_words if w in page_word_set]

    # Defensive cleanup (prevents junk leaking)
    matched = [w for w in matched if len(w) >= 4 and w not in STOPWORDS]

    # Score by matches count (capped)
    score = min(len(set(matched)) * 20, 100)

    # Deduplicate while keeping order
    seen = set()
    matched_ordered = []
    for w in matched:
        if w not in seen:
            matched_ordered.append(w)
            seen.add(w)

    return score, matched_ordered


def build_relevance_reason(title: str, url: str, summary: str, profile: str, matched_words: List[str]) -> str:
    """
    Human, professional reason (no 'on', 'and', etc.).
    """
    top_terms = matched_words[:5]

    # SGI-style: tie topic + role interest
    if top_terms:
        return (
            f"This is relevant to you because it focuses on {', '.join(top_terms)}, "
            f"which aligns with the interests described in your profile."
        )

    # Fallback
    return "This item broadly aligns with themes described in your profile."


# --------------------------
# LINK DISCOVERY (FROM HOMEPAGE)
# --------------------------
def extract_internal_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    """
    Pull internal links from homepage, including deep article links.
    Filters out fragments, non-domain links, and obvious auth/subscribe pages.
    """
    links = set()
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("#"):
            continue

        full = urljoin(base_url, href)
        parsed = urlparse(full)

        if parsed.netloc != base_domain:
            continue

        # skip obvious non-content
        lower_path = (parsed.path or "").lower()
        if any(x in lower_path for x in ["/account", "/login", "/signin", "/subscribe", "/checkout"]):
            continue

        # normalize (remove fragments if any got through)
        cleaned = full.split("#")[0].rstrip("/")
        if not cleaned:
            continue

        # keep
        links.add(cleaned)

    # Prefer article URLs first (so we fetch more useful pages)
    links_list = list(links)
    links_list.sort(key=lambda u: (not is_article_url(u), is_hub_or_index(u), len(u)))
    return links_list


# --------------------------
# FETCH + ANALYZE A SINGLE PAGE
# --------------------------
async def fetch_and_analyze(client: httpx.AsyncClient, url: str, profile: str) -> Optional[dict]:
    try:
        resp = await client.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    title = extract_title(soup)
    text = extract_main_text(soup)

    if len(text) < MIN_TEXT_LEN:
        return None

    # Create a short summary (first ~2–3 sentences from extracted text)
    summary = text[:900]
    summary = normalize_spaces(summary)

    score, matched = score_relevance(text, profile)

    if score == 0:
        return None

    # Hub/index pages must be extremely relevant
    if (not is_article_url(url)) and is_hub_or_index(url) and score < HUB_SCORE_THRESHOLD:
        return None

    # Clean relevance reason (no junk words)
    reason = build_relevance_reason(title, url, summary, profile, matched)

    return {
        "title": title,
        "url": url,
        "summary": summary,
        "relevance_score": score,
        "relevance_reason": reason,
    }


# --------------------------
# MAIN RECOMMENDER
# --------------------------
async def recommend_from_sgi(profile: str) -> List[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGI-Recommender/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        # 1) Fetch homepage
        homepage = await client.get(BASE_SITE, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(homepage.text, "html.parser")

        # 2) Collect internal links
        links = extract_internal_links(BASE_SITE, soup)

        # Limit total links we attempt
        links = links[:MAX_LINKS_FROM_HOME]

        # 3) Fetch & analyze pages (bounded)
        links_to_fetch = links[:MAX_PAGES_TO_FETCH]

        tasks = [fetch_and_analyze(client, u, profile) for u in links_to_fetch]
        results = await asyncio.gather(*tasks)

        items = [r for r in results if r]

        # 4) Sort by score, and prefer articles over hubs when tied
        items.sort(
            key=lambda x: (
                x.get("relevance_score", 0),
                1 if is_article_url(x.get("url", "")) else 0,
                -len(x.get("summary", "")),
            ),
            reverse=True,
        )

        # 5) Return top N
        return items[:TOP_RESULTS]


# --------------------------
# HEALTH + API
# --------------------------
@app.get("/")
def health():
    return {"status": "SGI Europe Recommender is running", "site": BASE_SITE}


@app.post("/recommend")
async def recommend(req: ProfileRequest):
    profile = normalize_spaces(req.profile)
    if not profile or len(profile) < 10:
        return {
            "articles": [],
            "error": "Please provide a longer profile (at least ~10 characters)."
        }

    try:
        items = await recommend_from_sgi(profile)
        return {"articles": items, "error": None}
    except Exception as e:
        return {"articles": [], "error": str(e)}
