import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

def fetch_html_content(url):
    """Fetches the HTML Content from a given URL with a user-agent to mimic a browser"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_text_from_html(html_content, base_url):
    """
    Parses HTML and extracts clean text from all visible elements,
    including content from common Google Sites blocks and embedded media.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_texts = []
    
    main_content_div = soup.find('div', class_='f32l6')
    if main_content_div:
        text_containers = main_content_div.find_all(['p', 'div', 'span', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table'])
        for container in text_containers:
            text = container.get_text(strip=True)
            if text:
                extracted_texts.append(text)

        iframes = main_content_div.find_all('iframe')
        for iframe in iframes:
            src = iframe.get('src')
            if not src:
                continue

            absolute_url = urljoin(base_url, src)

            # Handle embedded Google Docs
            if "docs.google.com/document" in src:
                match = re.search(r'/d/e/(.*?)/', absolute_url)
                if match:
                    doc_id = match.group(1)
                    export_url = f"https://docs.google.com/document/d/e/{doc_id}/export?format=html"
                    embedded_html = fetch_html_content(export_url)
                    if embedded_html:
                        embedded_soup = BeautifulSoup(embedded_html, 'html.parser')
                        extracted_texts.append(embedded_soup.get_text(separator=' ', strip=True))

            # Handle embedded Google Sheets
            elif "docs.google.com/spreadsheets" in src:
                match = re.search(r'/d/e/([a-zA-Z0-9-_]+)', absolute_url)
                if match:
                    sheet_id = match.group(1)
                    export_url = f"https://docs.google.com/spreadsheets/d/e/{sheet_id}/pub?output=csv"
                    csv_data = fetch_html_content(export_url)
                    if csv_data:
                        rows = csv_data.splitlines()
                        for row in rows:
                            extracted_texts.append(row)

            else:
                extracted_texts.append(f"Embedded content detected. Source URL: {absolute_url}")


        images = main_content_div.find_all('img')
        for img in images:
            alt = img.get('alt')
            title = img.get('title')
            if alt:
                extracted_texts.append(f"Image alt text: {alt}")
            if title:
                extracted_texts.append(f"Image title: {title}")
    else:
        extracted_texts.append(soup.body.get_text(separator=' ', strip=True))

    return ' '.join(extracted_texts)

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks respecting sentence boundaries.
    """
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?]) +', text.strip())

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            
            sentences_for_overlap = ""
            current_length = 0
            for s in reversed(current_chunk.split('. ')):
                if current_length + len(s) + 1 <= overlap:
                    sentences_for_overlap = s + ". " + sentences_for_overlap
                    current_length += len(s) + 1
                else:
                    break
            
            current_chunk = sentences_for_overlap.strip() + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_data_records(url, text_chunks, course_id, doc_type="Google_Site"):
    """Combines text chunks with metadata to create a list of data records."""
    data_records = []
    for i, chunk in enumerate(text_chunks):
        record = {
            'text': chunk,
            'metadata': {
                'course_id': course_id,
                'source_url': url,
                'chunk_id': f"{course_id}_{i}",
                'document_type': doc_type
            }
        }
        data_records.append(record)
    return data_records