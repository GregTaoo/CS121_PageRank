import certifi
import requests
from bs4 import BeautifulSoup
from collections import deque
import time
import urllib.parse
import posixpath

START_URL = "https://sist.shanghaitech.edu.cn/"
MAX_PAGES = 100000
DELAY = 0.1
OUTPUT_FILE = "web_graph.txt"
MAP_FILE = "url_map.txt"

EXCLUDED_EXTENSIONS = (
    '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
    '.pdf', '.doc', '.docx', '.ppt', '.zip', '.rar', '.exe', '.mp4', '.mp3'
)

class WebGraphCrawler:
    def __init__(self, start_url):
        self.start_url = self.normalize_url(start_url, start_url)
        self.url_to_id = {}
        self.id_to_url = {}
        self.next_id = 0
        self.visited = set()
        self.queue = deque()
        self.edges = []

        self.get_id(self.start_url)
        self.queue.append(self.start_url)

    def get_id(self, url):
        if url not in self.url_to_id:
            curr_id = self.next_id
            self.url_to_id[url] = curr_id
            self.id_to_url[curr_id] = url
            self.next_id += 1
            return curr_id
        return self.url_to_id[url]

    def is_valid_url(self, url):
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme not in ['http', 'https']:
            return False

        if url.find('shanghaitech') == -1:
            return False

        path = parsed.path.lower()
        if path.endswith(EXCLUDED_EXTENSIONS):
            return False

        return True

    def normalize_url(self, base_url, href):
        href = href.strip()
        if not href:
            return None

        if href.startswith(('javascript:', 'mailto:', '#')):
            return None

        base_parsed = urllib.parse.urlparse(base_url)
        site_root = f"{base_parsed.scheme}://{base_parsed.netloc}/"

        full_url = urllib.parse.urljoin(site_root, href)
        full_url = full_url.split('#')[0]

        parsed = urllib.parse.urlparse(full_url)

        norm_path = posixpath.normpath(parsed.path)
        if not norm_path.startswith('/'):
            norm_path = '/' + norm_path

        normalized = f"{parsed.scheme}://{parsed.netloc}{norm_path}"

        return normalized

    def fetch_links(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                verify=False,
                timeout=5
            )

            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            links = set()

            for tag in soup.find_all('a', href=True):
                full_url = self.normalize_url(url, tag['href'])
                if not full_url:
                    continue

                if urllib.parse.urlparse(full_url).path.count('/') > 12:
                    continue

                if self.is_valid_url(full_url):
                    links.add(full_url)

            return list(links)

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []

    def run(self):
        print(f"Starting from: {self.start_url}")

        crawled_count = 0

        while self.queue and crawled_count < MAX_PAGES:
            current_url = self.queue.popleft()

            if current_url in self.visited:
                continue

            print(current_url)
            src_id = self.get_id(current_url)
            self.visited.add(current_url)
            crawled_count += 1

            child_urls = self.fetch_links(current_url)

            for child_url in child_urls:
                dst_id = self.get_id(child_url)
                self.edges.append((src_id, dst_id))

                if child_url not in self.visited:
                    self.queue.append(child_url)

        self.save_results()

    def save_results(self):
        with open(OUTPUT_FILE, 'w') as f:
            f.write(f"{self.next_id} {len(self.edges)} \n")
            for u, v in self.edges:
                f.write(f"{u} {v} 1\n")

        with open(MAP_FILE, 'w', encoding='utf-8') as f:
            for i in range(self.next_id):
                f.write(f"{i} {self.id_to_url[i]}\n")

        print(f"V: {self.next_id}, E: {len(self.edges)}")

if __name__ == "__main__":
    crawler = WebGraphCrawler(START_URL)
    crawler.run()
