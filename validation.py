"""
Standalone Timetable Web Scraper Module
===================================================================
This module constructs a direct search URL for IDOS.cz, downloads the
raw webpage HTML, and extracts the scheduled travel time using BeautifulSoup.

It requires the 'beautifulsoup4' and 'requests' libraries.
Install them via: pip install requests beautifulsoup4
"""
import requests
from bs4 import BeautifulSoup
import time
import re

# Memory cache to prevent spamming IDOS with duplicate segment lookups
_segment_cache = {}


def get_scraped_segment_time(origin_station: str, dest_station: str) -> float | None:
    """
    Scrapes IDOS.cz for the scheduled duration between two stations.

    Args:
        origin_station (str): The departure station name.
        dest_station (str): The arrival station name.

    Returns:
        float: The scheduled time in seconds, or None if scraping fails.
    """
    # Clean the station names slightly to help the IDOS search engine
    origin_clean = origin_station.replace(" z.", "").replace(" zast.", "").strip()
    dest_clean = dest_station.replace(" z.", "").replace(" zast.", "").strip()

    cache_key = f"{origin_clean}_{dest_clean}"
    if cache_key in _segment_cache:
        return _segment_cache[cache_key]

    # Target the IDOS trains-only endpoint
    url = "https://idos.idnes.cz/vlaky/spojeni/vysledky/"

    # We must use a standard browser User-Agent, otherwise IDOS will block the request as a bot
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "cs,en;q=0.9"
    }

    # direct=true forces a single train without transfers
    params = {
        "f": origin_clean,
        "t": dest_clean,
        "direct": "true"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10.0)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # IDOS typically stores the duration in a specific paragraph class
            # Example: <p class="reset duration"><strong>15 min</strong></p>
            duration_tags = soup.find_all("p", class_="reset duration")

            for tag in duration_tags:
                text = tag.get_text().strip()

                # Use Regex to extract hours and minutes (e.g. "1 hod 15 min" or "15 min")
                hours = 0
                minutes = 0

                h_match = re.search(r'(\d+)\s*hod', text)
                if h_match:
                    hours = int(h_match.group(1))

                m_match = re.search(r'(\d+)\s*min', text)
                if m_match:
                    minutes = int(m_match.group(1))

                if hours > 0 or minutes > 0:
                    delta_seconds = float((hours * 3600) + (minutes * 60))

                    _segment_cache[cache_key] = delta_seconds

                    # Be polite to the server and pause briefly
                    time.sleep(0.5)
                    return delta_seconds

            print(f"Scraper: No duration found on page for {origin_clean} -> {dest_clean}.")
            return None

        else:
            print(f"HTTP Error {response.status_code} for {origin_clean} -> {dest_clean}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Scraping Network Error: {e}")
        return None
    except Exception as e:
        print(f"Parsing Error: {e}")
        return None


# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing IDOS Web Scraper...")

    # Test a known segment
    test_origin = "Ledečko"
    test_dest = "Samopše"

    print(f"Searching for connection: {test_origin} -> {test_dest}")
    test_time = get_scraped_segment_time(test_origin, test_dest)

    if test_time:
        print(f"Success! Scheduled time found: {test_time} seconds ({test_time / 60:.1f} minutes).")
    else:
        print("Failed to scrape schedule.")