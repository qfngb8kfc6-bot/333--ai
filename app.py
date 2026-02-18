from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class RecommendRequest(BaseModel):
    url: str


@app.post("/recommend")
def recommend(data: RecommendRequest):
    try:
        # Fetch website
        response = requests.get(data.url, timeout=10)
        response.raise_for_status()

        # Parse content
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts/styles
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)
        text = text[:4000]  # Limit for token safety

        # AI Analysis
        ai_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a business intelligence AI that analyzes websites and extracts relevant content."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the following website content.

1. Extract the most relevant content from the site.
2. Explain why this content is important.
3. Give reasoning and guidance tailored to a potential client.

Website Content:
{text}
"""
                }
            ],
            temperature=0.7,
        )

        return {
            "response": ai_response.choices[0].message["content"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
