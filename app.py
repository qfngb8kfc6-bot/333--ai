# app.py
import os
import re
import json
import traceback
from typing import List, Optional
from urllib.parse import urlparse, urljoin

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Optional HTML parsing (recommended) ----
# If BeautifulSoup is not installed, we still work (we just use raw text).
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

# ---- HTTP client ----
import httpx

# ---- OpenAI ----
# Works with openai>=1.x
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# Config
# =========================
APP_NAME = os.getenv("APP_NAME", "Client-to-Host Relevance Widget API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # you can override on Render
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "10"))

# Allow all origins by default (easy for embedding)
# If you want to lock it down later, set ALLOWED_ORIGINS to comma-separated list.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

# Render/hosting sometimes blocks aggressive bots; set a polite UA.
DEFAULT_UA = os.getenv(
    "SCRAPE_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)

# Instantiate OpenAI client lazily
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not OpenAI:
        raise HTTPException(
            status_code=500,
            detail="OpenAI python package is not installed. Add 'openai' to requirements.txt.",
        )
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the server.")
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# =========================
# Models
# =========================
class RecommendRequest(BaseModel):
    clientRole: str = Field(..., description="Role of the client user (e.g., 'Founder', 'Marketing Manager').")
    clientCompany: str = Field(..., description="Company name for context (optional but helpful).")
    clientUrl: str = Field(..., description="URL entered by the end-user (their site).")
    hostUrl: str = Field(..., description="Host/site URL where the widget is embedded (your site).")
    maxHighlights: int = Field(6, ge=1, le=12, description="How many recommendations to return.")


class Highlight(BaseModel):
    title: str
    url: str
    snippet: str


class RecommendResponse(BaseModel):
    hostName: str
    clientSummary: str
    hostSummary: str
    paragraph: str
    highlights: List[Highlight]


# =========================
# App
# =========================
app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Helpers
# =========================
def safe_hostname(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def extract_visible_text(html: str) -> str:
    """
    Extract readable text from HTML. Uses BeautifulSoup if available,
    otherwise falls back to stripping tags roughly.
    """
    if not html:
        return ""

    if BeautifulSoup is None:
        # crude fallback
        text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return normalize_text(text)[:8000]

    soup = BeautifulSoup(html, "html.parser")

    # remove junk
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return normalize_text(text)[:8000]


def extract_title_and_description(html: str) -> tuple[str, str]:
    if not html:
        return ("", "")
    if BeautifulSoup is None:
        return ("", "")

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    desc = ""

    if soup.title and soup.title.string:
        title = normalize_text(soup.title.string)

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = normalize_text(meta.get("content"))

    return (title, desc)


def extract_links(html: str, base_url: str, limit: int = 25) -> List[str]:
    """
    Extract a small set of likely-informational links from a page.
    """
    if not html or BeautifulSoup is None:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue

        # skip junk
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:") or "javascript:" in href:
            continue

        full = urljoin(base_url, href)
        # only http(s)
        if not full.startswith("http://") and not full.startswith("https://"):
            continue

        if full in seen:
            continue
        seen.add(full)

        links.append(full)
        if len(links) >= limit:
            break

    return links


async def fetch_url(url: str) -> str:
    """
    Fetch URL HTML. Raises HTTPException with a helpful message on failure.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"},
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.text
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL (HTTP {e.response.status_code}): {url}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=400, detail=f"Timed out fetching URL: {url}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {url} ({str(e)})")


async def llm_recommend(
    req: RecommendRequest,
    client_context: dict,
    host_context: dict,
    host_links: List[str],
) -> RecommendResponse:
    """
    Calls OpenAI to generate recommendations. Returns structured JSON.
    """
    client = get_openai_client()

    # Give the model enough context without dumping huge pages
    payload = {
        "client": {
            "role": req.clientRole,
            "company": req.clientCompany,
            "url": req.clientUrl,
            "title": client_context.get("title", ""),
            "description": client_context.get("description", ""),
            "summaryText": client_context.get("text", "")[:2500],
        },
        "host": {
            "url": req.hostUrl,
            "title": host_context.get("title", ""),
            "description": host_context.get("description", ""),
            "summaryText": host_context.get("text", "")[:2500],
            "links": host_links[:25],
        },
        "maxHighlights": req.maxHighlights,
    }

    system = (
        "You are an expert website assistant that recommends services/products from the host site to a client. "
        "You must return STRICT JSON matching the schema:\n"
        "{\n"
        '  "hostName": string,\n'
        '  "clientSummary": string,\n'
        '  "hostSummary": string,\n'
        '  "paragraph": string,\n'
        '  "highlights": [ { "title": string, "url": string, "snippet": string } ]\n'
        "}\n"
        "Rules:\n"
        "- Use host.links when possible for highlight URLs.\n"
        "- Highlights should be service-like recommendations.\n"
        "- Explain WHY each fits the client.\n"
        "- Keep snippets short (1–2 sentences).\n"
        "- Do not include any keys besides those required.\n"
    )

    user = (
        "Generate host service recommendations for this client using the provided context.\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""

        # Extract first JSON object in case the model adds extra text
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Model did not return JSON.")

        data = json.loads(match.group(0))

        # Validate to our response model (this also cleans types)
        return RecommendResponse(**data)

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to Render logs
        print("LLM ERROR:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Main endpoint the widget calls.
    """
    try:
        # 1) Fetch both pages
        client_html = await fetch_url(req.clientUrl)
        host_html = await fetch_url(req.hostUrl)

        # 2) Extract content
        client_title, client_desc = extract_title_and_description(client_html)
        host_title, host_desc = extract_title_and_description(host_html)

        client_text = extract_visible_text(client_html)
        host_text = extract_visible_text(host_html)

        host_links = extract_links(host_html, req.hostUrl, limit=30)

        client_context = {
            "title": client_title,
            "description": client_desc,
            "text": client_text,
        }
        host_context = {
            "title": host_title,
            "description": host_desc,
            "text": host_text,
        }

        # 3) Ask the LLM for recommendations
        response = await llm_recommend(req, client_context, host_context, host_links)

        # 4) Safety: ensure highlight list length
        response.highlights = response.highlights[: req.maxHighlights]
        return response

    except HTTPException:
        raise
    except Exception as e:
        print("RECOMMEND ERROR:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Render health checks sometimes hit "/"
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "openapi": "/openapi.json"}

