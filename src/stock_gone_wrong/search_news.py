"""Google search news in specified timeframe

Reference: https://github.com/Nv7-GitHub/googlesearch
"""

from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from googlesearch import SearchResult, get_useragent
from pandas import Timestamp


def request_news_with_consent(query: str, start: Timestamp, end: Timestamp, lang="en"):
    start_str = start.strftime("%m/%d/%Y")
    end_str = end.strftime("%m/%d/%Y")
    response = requests.get(
        url="https://www.google.com/search",
        headers={"User-Agent": get_useragent(), "Accept": "*/*"},
        params={
            "q": query,
            "tbm": "nws",
            "tbs": f"cdr:1,cd_min:{start_str},cd_max:{end_str}",
            "hl": lang,
        },
        timeout=5,
        cookies={
            "CONSENT": "PENDING+987",
            "SOCS": "CAESHAgBEhIaAB",
        },
    )

    return response


def scrape_search_results(soup: BeautifulSoup) -> list[SearchResult]:
    result_block = soup.find_all("div", class_="ezO2md")

    results: list[SearchResult] = []
    for result in result_block:
        link_tag = result.find("a", href=True)
        title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
        description_tag = result.find("span", class_="FrIlee")

        link = (
            unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
            if link_tag
            else ""
        )
        title = title_tag.text if title_tag else ""
        description = description_tag.text if description_tag else ""
        results.append(SearchResult(link, title, description))
    return results


def search_news(query: str, start: Timestamp, end: Timestamp, lang="en"):
    response = request_news_with_consent(query, start, end, lang)
    soup = BeautifulSoup(response.content, "html.parser")
    return scrape_search_results(soup)
