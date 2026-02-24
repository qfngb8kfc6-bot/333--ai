import asyncio
import re
from typing import List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_SITE = "https://www.sgieurope.com"
MAX_PAGES_TO_SCAN = 25
TOP_RESULTS = 5

# ==========================================================
# FASTAPI SETUP
# ==========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# REQUEST MODEL
# ==========================================================

class ProfileRequest(BaseModel):
    profile: str


# ==========================================================
# STOPWORDS & TEXT CLEANING
# ==========================================================

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from",
    "have", "has", "had", "was", "were", "are", "but",
    "not", "you", "your", "about", "into", "over",
    "under", "their", "there", "they", "them",
    "what", "when", "where", "which", "while",
    "who", "whom", "why", "how", "will", "would",
    "can", "could", "should", "may", "might",
    "been", "being", "also", "than", "then",
    "its", "it's", "our", "out", "off", "per",
    "via", "due", "all", "any", "each", "other",
    "on", "in", "at", "to", "of", "by", "as"
}


def clean_words(text: str):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


# ==========================================================
# RELEVANCE SCORING
# ==========================================================

def score_relevance(page_text: str, profile: str):
    profile_words = clean_words(profile)
    page_words = clean_words(page_text)

    page_word_set = set(page_words)

    matched_words = []
    score = 0

    for word in profile_words:
        if word in page_word_set:
            score += 20
            matched_words.append(word)

    score = min(score, 100)

    return score, list(set(matched_words))


# ==========================================================
# EXTRACT INTERNAL LINKS
# ==========================================================

def extract_internal_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    links = set()
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.startswith("#"):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.netloc != base_domain:
            continue

        if parsed.path in ["", "/"]:
            continue

        links.add(full_url)

    return list(links)


# ==========================================================
# SCRAPE SINGLE PAGE
# ==========================================================

async def scrape_page(client: httpx.AsyncClient, url: str, profile: str):
    try:
        response = await client.get(url, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else "Untitled"

        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs[:10]])

        if len(content.strip()) < 300:
            return None

        relevance_score, matched_words = score_relevance(content, profile)

        if relevance_score == 0:
            return None

        explanation = (
            f"This article is relevant because it discusses "
            f"{', '.join(matched_words[:5])}, "
            f"which aligns with the visitor's stated interests."
        )

        return {
            "title": title,
            "url": url,
            "summary": content[:700] + "...",
            "relevance_score": relevance_score,
            "relevance_reason": explanation
        }

    except Exception:
        return None


# ==========================================================
# MAIN SCRAPER LOGIC
# ==========================================================

async def scrape_sgi(profile: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:

        homepage = await client.get(BASE_SITE, timeout=10)
        soup = BeautifulSoup(homepage.text, "html.parser")

        links = extract_internal_links(BASE_SITE, soup)
        links = links[:MAX_PAGES_TO_SCAN]

        tasks = [
            scrape_page(client, link, profile)
            for link in links
        ]

        results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if r is not None]

        valid_results.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        return valid_results[:TOP_RESULTS]


# ==========================================================
# API ENDPOINT
# ==========================================================

@app.post("/recommend")
async def recommend(request: ProfileRequest):
    try:
        articles = await scrape_sgi(request.profile)

        return {
            "articles": articles,
            "error": None
        }

    except Exception as e:
        return {
            "articles": [],
            "error": str(e)
        }
