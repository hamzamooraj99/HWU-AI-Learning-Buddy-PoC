import requests
from bs4 import BeautifulSoup

def fetch_html_content(url):
    """Fetches the HTML Content from a given URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_text_from_html(html_content):
    """Parses HTML and extracts clean text"""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Example: Google Sites often use a 'div' with role="main" for the main content
    main_content_div = soup.find('div', {'role': 'main'})

    if main_content_div:
        return main_content_div.get_text(separator=' ', strip=True)
    else:
        # Fallback to get all text from the body if the main content div isn't found
        return soup.body.get_text(separator=' ', strip=True)
    
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of a specified size with an overlap.
    A simple but effective way to ensure context is not lost.
    """
    chunks = []
    if not text: 
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    
    return chunks

def create_data_records(url, text_chunks, course_id):
    """
    Combines text chunks with metadata to create a list of data records.
    """
    data_records = []
    for i, chunk in enumerate(text_chunks):
        record = {
            'text': chunk,
            'metadata': {
                'course_id': course_id,
                'source_url': url,
                'chunk_id': f"{course_id}_{i}",
                'document_type': "Google_Site"
            }
        }
        data_records.append(record)
    return data_records
    
