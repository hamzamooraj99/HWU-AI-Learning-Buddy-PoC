import os
import pathlib
import re
import json
import pymupdf4llm as pymu
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean up text by replacing multiple newlines and spaces with single ones."""
    if not text:
        return ""
    text = re.sub(r'\n{2,}', '\n', str(text))  # multiple newlines → one
    text = re.sub(r' {2,}', ' ', text)         # multiple spaces → one
    return text.strip()

def extract_md_from_pdf(pdf_path: str) -> str:
    """Convert a PDF file into Markdown text using pymupdf4llm"""
    try:
        md_text = pymu.to_markdown(pdf_path)
        return md_text
    except Exception as e:
        print(f"[ERR] Error converting {pdf_path} to Markdown: {e}")
        return

def split_md_by_headings(md_text: str) -> List[Dict[str, str]]:
    """
    Splits a markdown string into chunks based on headings and 
    returns a list of dictionaries with the content and its heading.
    """
    # Regex to find all markdown headings (e.g. # Heading, ## Subheading)
    # Regex to find any heading (from # to ######)
    heading_pattern = re.compile(r'^(#{1,6})\s*(.*)$', re.MULTILINE)

    # Split doc by headings
    matches = list(heading_pattern.finditer(md_text))

    chunks = []
    heading_hierarchy = {}

    # Handle preamble before the first heading
    start_index = 0
    if matches:
        start_index = matches[0].start()
    preamble = md_text[:start_index].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "metadata": {
                "heading": "Document Preamble",
                "heading_path": "Document Preamble"
            }
        })

    for i, match in enumerate(matches):
        heading_level = len(match.group(1))
        heading_title = match.group(2).strip()

        # Determine the content start/end
        content_start = match.end()
        content_end = matches[i+1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[content_start:content_end].strip()

        # Update heading hierarchy
        heading_hierarchy[heading_level] = heading_title
        # Clear deeper levels
        deeper_levels = [lvl for lvl in heading_hierarchy.keys() if lvl > heading_level]
        for lvl in deeper_levels:
            del heading_hierarchy[lvl]
        
        # Construct the full heading path
        heading_path = ">".join(heading_hierarchy.values())

        # Create the chunk
        chunks.append({
            "text": f"{match.group(0).strip()}\n{content}",
            "metadata": {
                "heading": heading_title,
                "heading_level": heading_level,
                "heading_path": heading_path
            }
        })
    
    # If no headings are found, treat the whole document as a single chunk
    if not matches and md_text.strip():
        chunks.append({
            "text": md_text.strip(),
            "metadata": {
                "heading": "Full Document", 
                "heading_path": "Full Document"
            }
        })
            
    return chunks

    
def process_pdf(docs_dir: str, course_id: str = "converted_markdown") -> str:
    """
    Processes all PDFs in a directory to Markdown using pymupdf4llm
    and saves the output to a JSON file.
    """
    final_data = []

    # Check if dir exists
    if not os.path.isdir(docs_dir):
        print(f"[ERROR] Directory not found: {docs_dir}")
        return
    
    # List all PDF files in the directory
    pdf_paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    
    if not pdf_paths:
        print(f"[WARN] No PDFs found in {docs_dir} - Nothing to do...")
        return
    
    for pdf_path in pdf_paths:
        print(f"[INFO] Processing {pdf_path}...")
        try:
            # Use pymu to convert the entire PDF to a single Markdown string
            md_text = pymu.to_markdown(pdf_path)

            # Split the md into smaller chunks
            chunks = split_md_by_headings(md_text)

            for chunk in chunks:
                final_data.append({
                    "text": chunk['text'],
                    "metadata": {
                        "course_id": course_id,
                        "source_path": pdf_path,
                        **chunk["metadata"]
                    }
                })
            
            print(f"[INFO] Successfully converted and split {pdf_path} into {len(chunks)} chunks")
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")
    
    if final_data:
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{course_id}_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Successfully created {output_file} with {len(final_data)} records.")
    else:
        print(f"[WARN] No records were generated for {course_id}.")

if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))

    course_ids = ["F21CA", "F21NL"]  # Define the courses to process
    for course_id in course_ids:
        course_docs_path = os.path.join(root_dir, "pdfs", (course_id.lower()))
        process_pdf(course_id=course_id, docs_dir=course_docs_path)