import os
import json
from llama_index.core import SimpleDirectoryReader, Document
from html_paraser import *

def main(course_id, url=None, pdf=None, docs_dir=None):
    """
    Main function to orchestrate the data ingestion process.
    """
    final_data = []

    if url:
        html = fetch_html_content(url=url)
        clean_text = extract_text_from_html(html)

        if not clean_text:
            print("Could not extract text...")
            return
        
        text_chunks = chunk_text(clean_text, chunk_size=2000, overlap=200)
        final_data = create_data_records(url, text_chunks, course_id)
    
    else:
        if pdf:
            reader = SimpleDirectoryReader(input_files=[pdf])
        elif docs_dir:
            reader = SimpleDirectoryReader(
                input_dir=docs_dir,
                required_exts=['.pdf'],
                recursive=True
            )
        documents = reader.load_data()
        for doc in documents:
            text_chunks = chunk_text(doc.text, chunk_size=2000, overlap=200)
            source_path = doc.metadata.get('file_path', 'unknown_source')
            final_data.extend(create_data_records(source_path, text_chunks, course_id))

    if final_data:
        output_dir = "data"
        output_file = os.path.join(output_dir, f"{course_id}_site_data.json")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    
    if __name__ == "__main__":
        main(course_id="F21CA", url="https://sites.google.com/view/test-f21ca/")