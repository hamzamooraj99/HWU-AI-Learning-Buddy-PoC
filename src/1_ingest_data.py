import os
import json
import re
import camelot
import fitz  # PyMuPDF
import pandas as pd


def clean_text(text: str) -> str:
    """Clean up text by replacing multiple newlines and spaces with single ones."""
    if not text:
        return ""
    text = re.sub(r'\n{2,}', '\n', str(text))  # multiple newlines → one
    text = re.sub(r' {2,}', ' ', text)         # multiple spaces → one
    return text.strip()


def extract_text_from_pdf(pdf_path: str, course_id: str) -> list[dict]:
    """Extract text from a PDF using PyMuPDF, return as a list of documents."""
    docs = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            docs.append({
                "text": clean_text(text),
                "metadata": {
                    "course_id": course_id,
                    "source_path": pdf_path,
                    "page": page_num,
                    "document_type": "PDF_Text"
                }
            })
    return docs


def convert_df_to_text(df: pd.DataFrame, pdf_path: str, table_idx: int, course_id: str) -> list[dict]:
    """
    Converts a pandas DataFrame into row-wise records in natural-ish language.
    """
    docs = []
    headers = [str(x) if x else f"col{j}" for j, x in enumerate(df.iloc[0])]
    for i, row in df.iloc[1:].iterrows():
        parts = []
        for j, value in enumerate(row):
            if pd.notna(value) and str(value).strip():
                parts.append(f"{headers[j]} is {clean_text(value)}")
        if parts:
            sentence = ". ".join(parts) + "."
            docs.append({
                "text": sentence,
                "metadata": {
                    "course_id": course_id,
                    "source_path": f"{pdf_path}_table_{table_idx}_row_{i}",
                    "document_type": "PDF_Table"
                }
            })
    return docs


def extract_tables_from_pdf(pdf_path: str, course_id: str) -> list[dict]:
    """Extract tables from PDF using Camelot, return as a list of documents."""
    docs = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")  # "lattice" if ruling lines
        for idx, table in enumerate(tables):
            df = table.df
            docs.extend(convert_df_to_text(df, pdf_path, idx, course_id))
    except Exception as e:
        print(f"[WARN] Could not extract tables from {pdf_path}: {e}")
    return docs


def process_pdfs(course_id: str, docs_dir: str, output_dir: str = "data"):
    """Main function to orchestrate PDF ingestion with PyMuPDF + Camelot."""
    final_data = []

    pdf_paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(".pdf")]

    for pdf_path in pdf_paths:
        print(f"[INFO] Processing {pdf_path} ...")

        # Extract text
        final_data.extend(extract_text_from_pdf(pdf_path, course_id))

        # Extract tables
        final_data.extend(extract_tables_from_pdf(pdf_path, course_id))

    # Save output
    if final_data:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{course_id}_site_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Successfully created {output_file} with {len(final_data)} records.")
    else:
        print(f"[WARN] No data extracted for {course_id}.")


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    course_ids = ["F21CA", "F21NL"]

    for course_id in course_ids:
        pdf_dir = os.path.abspath(os.path.join(this_dir, "..", "pdfs", course_id.lower()))
        process_pdfs(course_id, pdf_dir)
