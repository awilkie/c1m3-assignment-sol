# ================================
# Standard library imports
# ================================
import os
import xml.etree.ElementTree as ET

# ================================
# Third-party imports
# ================================
import requests
from duckduckgo_search import DDGS

session = requests.Session()
session.headers.update(
    {"User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"}
)


def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches arXiv for research papers matching the given query.
    """
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

    try:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
            ]
            published = entry.find("atom:published", ns).text[:10]
            url_abstract = entry.find("atom:id", ns).text
            summary = entry.find("atom:summary", ns).text.strip()

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "url": url_abstract,
                    "summary": summary,
                    "link_pdf": link_pdf,
                }
            )

        return results
    except Exception as e:
        return [{"error": f"Parsing failed: {str(e)}"}]


arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform a search using DuckDuckGo.

    Args:
        query (str): The search query.
        max_results (int): Number of results to return (default 5).

    Returns:
        list[dict]: A list of dictionaries with keys like 'title', 'content', and 'url'.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]

web_search_tool_def = {
    "type": "function",
    "function": {
        "name": "web_search_tool",
        "description": "Performs a general-purpose web search using DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def parse_input(text_or_messages):
    if isinstance(text_or_messages, list):
        text_report = None
        for m in reversed(text_or_messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            )
            if role == "assistant" and content:
                text_report = content
                break
        if not text_report:
            raise ValueError("No assistant text found in messages.")

    else:
        text_report = str(text_or_messages)

    return text_report
