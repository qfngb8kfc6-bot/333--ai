import asyncio
from typing import List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# -------------------------
# CORS (ALLOW ALL FOR NOW)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Model
# -------------------------
class RecommendRequest(BaseModel):
    url: str
    client_niche: str


# -------------------------
# Extract Article Links
# -------------------------
def extract_article_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    article_links = set()
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Skip fragments
        if "#" in href:
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Keep only same domain
        if parsed.netloc != base_domain:
            continue

        path_parts = parsed.path.strip("/").split("/")

        # Must have at least 2 segments (avoid homepage + categories)
        if len(path_parts) < 2:
            continue

        # Skip short paths (usually categories)
        if len(parsed.path) < 20:
            continue

        article_links.add(full_url)

    return list(article_links)


# -------------------------
# Score Relevance
# -------------------------
def score_relevance(text: str, niche: str) -> int:
    niche_words = niche.lower().split()
    score = 0

    for word in niche_words:
        if word in text.lower():
            score += 20

    return min(score, 100)


# -------------------------
# Scrape Single Article
# -------------------------
async def scrape_article(client: httpx.AsyncClient, url: str, niche: str):
    try:
        response = await client.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else "No title"

        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs[:5]])

        relevance_score = score_relevance(content, niche)

        return {
            "title": title,
            "url": url,
            "summary": content[:500],
            "relevance_score": relevance_score,
            "relevance_reason": f"Matched keywords related to '{niche}'."
            if relevance_score > 0
            else "Low keyword overlap with niche.",
        }

    except Exception:
        return None


# -------------------------
# Scrape Articles Async
# -------------------------
async def scrape_articles_async(base_url: str, niche: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        article_links = extract_article_links(base_url, soup)

        tasks = [
            scrape_article(client, link, niche)
            for link in article_links[:10]  # limit to first 10
        ]

        results = await asyncio.gather(*tasks)

        articles = [r for r in results if r is not None]

        # Sort by relevance
        articles.sort(key=lambda x: x["relevance_score"], reverse=True)

        return articles[:5]


# -------------------------
# API Endpoint
# -------------------------
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        url = request.url.strip()

        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        articles = await scrape_articles_async(url, request.client_niche)

        return {
            "articles": articles,
            "error": None
        }

    except Exception as e:
        return {
            "articles": [],
            "error": str(e)
        }


    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
