import os
import json
from llama_index.core import SimpleDirectoryReader, Document
import camelot
from html_parser import *

def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="all")
    return [t.df for t in tables]

def df_to_json_records(df, source="table"):
    records = []
    for _, row in df.iterrows():
        record = {col: str(val) for col, val in row.items()}
        records.append({
            "source": source,
            "content": str(record),   # store as text for embedding
            "metadata": record        # keep structured data too
        })
    return records

def main(course_id, url=None, pdf=None, docs_dir=None):
    """
    Main function to orchestrate the data ingestion process.
    """
    final_data = []

    if url:
        html = fetch_html_content(url=url)
        clean_text = extract_text_from_html(html, url)

        if not clean_text:
            print("Could not extract text...")
            return
        
        text_chunks = chunk_text(clean_text, chunk_size=2000, overlap=200)
        final_data = create_data_records(url, text_chunks, course_id)
    
    else:
        if pdf:
            pdf = os.path.abspath(pdf)
            print(f"[DEBUG] Looking for PDF at: {pdf}")
            if not os.path.exists(pdf):
                print("[ERROR] PDF file not found!")
                return
            reader = SimpleDirectoryReader(input_files=[pdf])
        elif docs_dir:
            docs_dir = os.path.abspath(docs_dir)
            print(f"[DEBUG] Looking for PDFs in dir: {docs_dir}")
            reader = SimpleDirectoryReader(
                input_dir=docs_dir,
                required_exts=['.pdf'],
                recursive=True
            )
        else:
            print("[ERROR] No valid input (pdf/docs_dir/url) provided")
            return

        documents = reader.load_data()
        print(f"[DEBUG] Loaded {len(documents)} document(s) from PDF(s)")

        for doc in documents:
            source_path = doc.metadata.get('file_path', pdf if pdf else docs_dir)
            text_chunks = chunk_text(doc.text, chunk_size=2000, overlap=200)
            final_data.extend(create_data_records(source_path, text_chunks, course_id, doc_type="PDF_Text"))
            try:
                tables = camelot.read_pdf(source_path, pages='all')
                for i, table in enumerate(tables):
                    df = table.df
                    for row_idx, row in df.iterrows():
                        record = {col: str(val) for col, val in row.items()}
                        final_data.append({
                            "text": str(record),
                            "metadata": {
                                "course_id": course_id,
                                "source_path": f"{source_path}#table{i}_row{row_idx}",
                                "document_type": "PDF_Table",
                                **record
                            }
                        })
                print(f"[INFO] Extracted {len(tables)} tables from {source_path}")
            except Exception as e:  
                print(f"[WARN] Could not extract tables from {source_path}: {e}")


    if final_data:
        output_dir = "data"
        output_file = os.path.join(output_dir, f"{course_id}_site_data.json")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print("UPDATED")
        
    
if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    pdf_path = os.path.abspath(os.path.join(this_dir, "..", "pdfs", "f21ca_site_text.pdf"))
    main(course_id="F21CA", pdf=pdf_path)