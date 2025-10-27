import requests
from bs4 import BeautifulSoup
from pathlib import Path

urls = [
    # Add Links to scrape here
]

# Save directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "cci_data.txt"

# Scrape and write content
with open(output_file, "w", encoding="utf-8") as f:
    for url in urls:
        print(f"Scraping: {url}")
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

        f.write(f"\n\n===== {url} =====\n")
        for para in paragraphs:
            f.write(para + "\n")

print(f"All pages scraped! Data saved to: {output_file}")
