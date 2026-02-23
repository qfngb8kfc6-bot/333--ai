import os
import re
import asyncio
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


# -----------------------------------
# Setup
# -----------------------------------

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY is missing!")

client = OpenAI(api_key=OPENAI_API_KEY)


class URLRequest(BaseModel):
    url: str


# -----------------------------------
# AI Summarizer
# -----------------------------------

async def summarize_with_ai(text: str) -> str:
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",  # cheaper + fast
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this article in 2 concise sentences for a business audience."
                },
                {
                    "role": "user",
                    "content": text[:4000]
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI summary failed: {str(e)}"


# -----------------------------------
# Async Article Fetcher
# -----------------------------------

async def fetch_article(client_http, url):
    try:
        resp = await client_http.get(url, timeout=10)
        resp.raise_for_status()
    except:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "No title"

    content_text = ""
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 80:
            content_text += text + " "

    if not content_text:
        return None

    ai_summary = await summarize_with_ai(content_text)

    return {
        "title": title,
        "url": url,
        "ai_summary": ai_summary
    }


# -----------------------------------
# Main Scraper
# -----------------------------------

async def scrape_articles_async(base_url, max_articles=5):

    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url.strip()

    parsed_base = urlparse(base_url)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    async with httpx.AsyncClient(headers=headers) as client_http:
        try:
            resp = await client_http.get(base_url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            return {"articles": [], "error": f"Failed to fetch site: {str(e)}"}

        soup = BeautifulSoup(resp.text, "html.parser")

        article_links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)

            parsed_full = urlparse(full_url)

            if parsed_full.netloc != parsed_base.netloc:
                continue

            if re.search(r"(news|article|blog|/20\d{2}/)", full_url.lower()):
                article_links.add(full_url)

            if len(article_links) >= max_articles:
                break

        tasks = [
            fetch_article(client_http, url)
            for url in list(article_links)[:max_articles]
        ]

        results = await asyncio.gather(*tasks)

        articles = [r for r in results if r is not None]

        return {
            "articles": articles,
            "error": None
        }


# -----------------------------------
# API Endpoint
# -----------------------------------

@app.post("/recommend")
async def recommend(data: URLRequest):
    result = await scrape_articles_async(data.url)

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
