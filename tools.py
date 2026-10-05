import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from rich import print

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic.

    Args:
        query: The search text to look up, for example "latest advances in quantum computing".
            This argument is required.
    """
    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY is not set. Add it to your .env file to enable web search."

    try:
        resp = requests.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "max_results": 4},
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        return f"Search failed: {e}"

    out = []
    for r in results.get("results", []):
        out.append(
            f"title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('content', '')[:250]}"
        )

    if not out:
        return "No results found."
    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape a web page and return its text content.

    Args:
        url: The full page address starting with http:// or https://. This argument is required.
    """
    try:
        resp = requests.get(url, timeout=8, headers={"user-agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:2000]
    except Exception as e:
        print(f"Error occurred while scraping {url}: {e}")
        return f"Error occurred while scraping the URL: {e}"
