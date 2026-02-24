import asyncio
from typing import List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_SITE = "https://www.sgieurope.com"
MAX_PAGES_TO_SCAN = 20
TOP_RESULTS = 5

app = FastAPI()

# --------------------------------------------------
# CORS (Allow widget access)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class ProfileRequest(BaseModel):
    profile: str


# --------------------------------------------------
# Extract Internal Links From SGI Europe
# --------------------------------------------------

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

        # Keep only SGI Europe domain
        if parsed.netloc != base_domain:
            continue

        # Remove homepage
        if parsed.path in ["", "/"]:
            continue

        links.add(full_url)

    return list(links)


# --------------------------------------------------
# Simple Keyword Relevance Scoring
# --------------------------------------------------

def score_relevance(page_text: str, profile: str):
    profile_words = profile.lower().split()
    text_lower = page_text.lower()

    score = 0
    matched_words = []

    for word in profile_words:
        if word in text_lower:
            score += 15
            matched_words.append(word)

    score = min(score, 100)

    return score, matched_words


# --------------------------------------------------
# Scrape Individual Page
# --------------------------------------------------

async def scrape_page(client: httpx.AsyncClient, url: str, profile: str):
    try:
        response = await client.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else "Untitled"

        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs[:8]])

        if len(content.strip()) < 200:
            return None

        relevance_score, matched_words = score_relevance(content, profile)

        if relevance_score == 0:
            return None

        return {
            "title": title,
            "url": url,
            "summary": content[:600],
            "relevance_score": relevance_score,
            "relevance_reason": (
                f"This page is relevant because it mentions: "
                f"{', '.join(set(matched_words))}."
            )
        }

    except Exception:
        return None


# --------------------------------------------------
# Main Scraper Logic
# --------------------------------------------------

async def scrape_sgi_europe(profile: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:

        # Get homepage
        response = await client.get(BASE_SITE, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        links = extract_internal_links(BASE_SITE, soup)

        # Limit number of pages scanned
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


# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/recommend")
async def recommend(request: ProfileRequest):
    try:
        results = await scrape_sgi_europe(request.profile)

        return {
            "articles": results,
            "error": None
        }

    except Exception as e:
        return {
            "articles": [],
            "error": str(e)
        }
