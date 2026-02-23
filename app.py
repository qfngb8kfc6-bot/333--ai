import os
import re
import asyncio
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


# -------------------------
# Setup
# -------------------------

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY is missing!")

client = OpenAI(api_key=OPENAI_API_KEY)


class URLRequest(BaseModel):
    url: str
    client_niche: str


# -------------------------
# AI Relevance Scorer
# -------------------------

async def score_relevance(niche: str, title: str, summary: str):

    prompt = f"""
    Client niche: {niche}

    Article:
    Title: {title}
    Summary: {summary}

    Determine:
    1) Relevance score from 0-100
    2) Short explanation (1 sentence)

    Return JSON only:
    {{
        "score": number,
        "reason": "string"
    }}
    """

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content

        # crude JSON parsing safety
        import json
        parsed = json.loads(content)

        return parsed["score"], parsed["reason"]

    except:
        return 0, "Scoring failed"


# -------------------------
# AI Summarizer
# -------------------------

async def summarize_with_ai(text: str):

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this article in 2 concise sentences for business professionals."
                },
                {
                    "role": "user",
                    "content": text[:3500]
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except:
        return "Summary unavailable"


# -------------------------
# Article Fetcher
# -------------------------

async def fetch_article(client_http, url, niche):

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

    summary = await summarize_with_ai(content_text)

    score, reason = await score_relevance(niche, title, summary)

    if score < 60:
        return None

    return {
        "title": title,
        "url": url,
        "summary": summary,
        "relevance_score": score,
        "relevance_reason": reason
    }


# -------------------------
# Main Scraper
# -------------------------

async def scrape_articles_async(base_url, niche, max_articles=8):

    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url.strip()

    parsed_base = urlparse(base_url)

    headers = {"User-Agent": "Mozilla/5.0"}

    async with httpx.AsyncClient(headers=headers) as client_http:

        try:
            resp = await client_http.get(base_url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            return {"articles": [], "error": str(e)}

        soup = BeautifulSoup(resp.text, "html.parser")

        links = set()

        for a in soup.find_all("a", href=True):
            full_url = urljoin(base_url, a["href"])
            parsed_full = urlparse(full_url)

            if parsed_full.netloc != parsed_base.netloc:
                continue

            if re.search(r"(news|article|blog|/20\d{2}/)", full_url.lower()):
                links.add(full_url)

            if len(links) >= max_articles:
                break

        tasks = [
            fetch_article(client_http, url, niche)
            for url in list(links)
        ]

        results = await asyncio.gather(*tasks)

        articles = [r for r in results if r]

        # sort by relevance score
        articles.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "articles": articles,
            "error": None
        }


# -------------------------
# API Endpoint
# -------------------------

@app.post("/recommend")
async def recommend(data: URLRequest):

    result = await scrape_articles_async(
        data.url,
        data.client_niche
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
