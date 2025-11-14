# rag_pipeline.py
import re
import os
import tempfile
from typing import List, Dict, Optional

from pypdf import PdfReader
from dotenv import load_dotenv
import anthropic

from weaviate_client import client
from schema import COLLECTION_NAME, ensure_schema, reset_collection

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages])


CASE_HEADER_PATTERN = r"(CASE\s+\d+:\s*.*)"  # e.g. "CASE 1: She Is Hungry"

def split_into_cases(text: str) -> List[Dict]:
    """
    Split the long document into individual case blocks.
    Returns [{'title': ..., 'body': ...}, ...]
    """
    sections = re.split(CASE_HEADER_PATTERN, text)
    cases = []

    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        cases.append({"title": title, "body": body})

    return cases