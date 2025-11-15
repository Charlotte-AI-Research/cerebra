from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

MAX_DEPTH = 25
MAX_FILE_SIZE_MB = 0.5  # rotate files when they exceed this size

# Starting URLs (can be any domain)
program_urls = [
    "https://cci.charlotte.edu"
]

# Derive base domain and file prefix from the first URL
parsed_root = urlparse(program_urls[0])
BASE_DOMAIN = f"{parsed_root.scheme}://{parsed_root.netloc}"
domain_prefix = parsed_root.netloc.replace(".", "_")

# Create the raw data folder
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

file_index = 1
current_file_path = output_dir / f"{domain_prefix}_{file_index}.txt"

visited = set()


def get_current_file():
    """Rotate files if current one exceeds size limit."""
    global file_index, current_file_path
    if current_file_path.exists():
        size_mb = current_file_path.stat().st_size / (1024 * 1024)
        if size_mb >= MAX_FILE_SIZE_MB:
            file_index += 1
            current_file_path = output_dir / f"{domain_prefix}_{file_index}.txt"
    return open(current_file_path, "a", encoding="utf-8")


def scrape_page(url, depth=0):
    """Scrape text from pages and follow internal links."""
    if url in visited or depth > MAX_DEPTH:
        return
    visited.add(url)

    print(f"[Depth {depth}] Scraping: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else "No Title"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

        # Write scraped text
        with get_current_file() as f:
            f.write(f"\n=== PAGE: {url} (Depth {depth}) ===\n")
            f.write(f"TITLE: {title}\n\n")
            for para in paragraphs:
                f.write(para + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

        # Recursively follow internal links on the same domain
        if depth < MAX_DEPTH:
            for link_tag in soup.find_all("a", href=True):
                full_url = urljoin(url, link_tag["href"])
                if full_url.startswith(BASE_DOMAIN):
                    scrape_page(full_url, depth + 1)

    except Exception as e:
        print(f" Error scraping {url}: {e}")


for url in program_urls:
    if url.startswith(BASE_DOMAIN):
        scrape_page(url)

print(
    f" Scraping complete! Files saved in '{output_dir}' "
    f"as {domain_prefix}_1.txt, {domain_prefix}_2.txt, etc."
)