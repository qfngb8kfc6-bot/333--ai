from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

app = FastAPI()

# ✅ CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qfngb8kfc6-bot.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI client securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RecommendRequest(BaseModel):
    url: HttpUrl


@app.get("/")
def health_check():
    return {"status": "AI Website Analyzer is running"}


@app.post("/recommend")
def recommend(data: RecommendRequest):
    try:
        # Fetch website
        response = requests.get(str(data.url), timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts/styles
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Extract clean text
        text = soup.get_text(separator=" ", strip=True)

        # Limit size (important for token limits)
        text = text[:6000]

        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable content found on site.")

        # AI Analysis
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a website intelligence AI that extracts meaningful business insights."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the following website content.

Return your answer in 3 sections:

1. Relevant Content Found
2. Why This Content Is Important
3. Tailored Guidance For A Potential Client

Website Content:
{text}
"""
                }
            ],
            temperature=0.7,
        )

        return {
            "analysis": completion.choices[0].message.content
        }

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=400,
            detail="Failed to fetch website. Ensure URL includes https://"
        )

    except Exception:
        # Don't expose raw OpenAI errors in production
        raise HTTPException(
            status_code=500,
            detail="AI service temporarily unavailable."
        )
