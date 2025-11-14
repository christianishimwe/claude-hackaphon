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

# ---------- PDF → TEXT ----------

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


# ---------- TEXT → CASE CHUNKS ----------

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


def _normalize_heading_line(line: str) -> str:
    """
    Normalize a potential heading line by:
    - removing leading emojis/symbols
    - trimming a trailing colon
    - lowercasing
    """
    # remove leading non-letters (emojis, icons like ✖, ⭐, ✓)
    cleaned = re.sub(r"^[^A-Za-z]+", "", line).strip()
    # remove trailing colon
    cleaned = re.sub(r":\s*$", "", cleaned)
    return cleaned.lower()


def _scan_sections(body: str) -> Dict[str, List[str]]:
    """
    Robust parser that scans line-by-line to find sections and bullet items.
    Handles headings that may include emojis (e.g., '⭐ Tone Guidelines')
    and bullet variants (•, -, –, —, *, ·, numbered lists).
    """
    target_map = {
        "forbidden words": "forbidden_words",
        "tone guidelines": "tone_guidelines",
        "good apology example structure": "example_structure",
    }

    sections: Dict[str, List[str]] = {
        "forbidden_words": [],
        "tone_guidelines": [],
        "example_structure": [],
    }

    current_key: Optional[str] = None

    # Bullet markers: -, •, ‣, ◦, ⁃, ∙, *, ·, –, —
    bullet_regex = re.compile(r"^\s*([-\u2022\u2023\u25E6\u2043\u2219\*\u00B7\u2013\u2014])\s+(.+?)\s*$")
    numbered_regex = re.compile(r"^\s*\d+[\.\)]\s+(.+?)\s*$")

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = _normalize_heading_line(line)
        if normalized in target_map:
            current_key = target_map[normalized]
            continue

        # Capture bullets only when inside a known section
        if current_key:
            m = bullet_regex.match(line)
            if m:
                sections[current_key].append(m.group(2))
                continue
            n = numbered_regex.match(line)
            if n:
                sections[current_key].append(n.group(1))
                continue

        # If it's a plain line directly under a section heading, treat it as content
        # (sometimes OCR drops bullet symbols). Only do this if it looks short like a bullet.
        if current_key and len(line) <= 140 and not line.endswith("."):
            sections[current_key].append(line)

    return sections


def parse_case(case_obj: Dict) -> Dict:
    """
    From a raw case body, try to pull:
    - forbidden_words
    - tone_guidelines
    - example_structure
    """
    body = case_obj["body"]

    # Primary path: robust line-by-line scanner
    sections = _scan_sections(body)

    forbidden_words = sections["forbidden_words"]
    tone_guidelines = sections["tone_guidelines"]
    example_structure = sections["example_structure"]

    return {
        "case_name": case_obj["title"],
        "forbidden_words": forbidden_words,
        "tone_guidelines": tone_guidelines,
        "example_structure": example_structure,
        "raw_body": body,
    }


# ---------- WEAVIATE INDEXING ----------

def index_cases(parsed_cases: List[Dict]) -> int:
    """
    Store parsed cases in Weaviate as objects in ApologyCase.
    """
    ensure_schema()
    coll = client.collections.get(COLLECTION_NAME)

    # Wipe existing rules to avoid mixing multiple docs
    reset_collection()

    for c in parsed_cases:
        # For retrieval we just need a single big text field
        chunk_text = (
            f"{c['case_name']}\n\n"
            f"Forbidden Words:\n" + "\n".join(c["forbidden_words"]) + "\n\n"
            f"Tone Guidelines:\n" + "\n".join(c["tone_guidelines"]) + "\n\n"
            f"Good Apology Example Structure:\n" + "\n".join(c["example_structure"]) + "\n\n"
            f"{c['raw_body']}"
        )

        coll.data.insert(
            {
                "case_name": c["case_name"],
                "raw_text": chunk_text,
            }
        )

    return len(parsed_cases)


def process_rules_pdf(uploaded_file) -> int:
    """
    High-level pipeline: FastAPI UploadFile → temp file → text → cases → parsed → index.
    """
    suffix = os.path.splitext(uploaded_file.filename)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = uploaded_file.file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path)
    os.remove(tmp_path)

    raw_cases = split_into_cases(text)
    parsed_cases = [parse_case(c) for c in raw_cases]

    count = index_cases(parsed_cases)
    return count


# ---------- RETRIEVAL + CLAUDE GENERATION ----------

def retrieve_best_case(case_description: str) -> Optional[Dict]:
    """
    Semantic search in Weaviate for the most relevant case.
    """
    ensure_schema()
    coll = client.collections.get(COLLECTION_NAME)

    result = coll.query.near_text(
        query=case_description,
        limit=1,
    )

    if not result.objects:
        return None

    obj = result.objects[0]
    return {
        "case_name": obj.properties["case_name"],
        "raw_text": obj.properties["raw_text"],
    }


def retrieve_relevant_rules(query: str, k: int = 2) -> List[Dict]:
    """
    Retrieve top-k relevant rule objects from Weaviate for a general question.
    Returns a list of dicts with case_name and raw_text.
    """
    ensure_schema()
    coll = client.collections.get(COLLECTION_NAME)

    result = coll.query.near_text(
        query=query,
        limit=max(1, k),
    )

    rules: List[Dict] = []
    if not result.objects:
        return rules

    for obj in result.objects:
        rules.append(
            {
                "case_name": obj.properties.get("case_name", ""),
                "raw_text": obj.properties.get("raw_text", ""),
            }
        )
    return rules


def generate_apology(case_description: str, wrongdoing: str) -> str:
    case_info = retrieve_best_case(case_description)

    if case_info is None:
        return (
            "I couldn't find any matching rules for that situation. "
            "Try describing the case differently."
        )

    rules_text = case_info["raw_text"]

    system_prompt = f"""
You are the 'Girlfriend Compliance Apology Generator.'

You are given a case rules document that contains:
- A case name
- Forbidden words and phrases
- Tone guidelines
- Example apology structure

You MUST:
- Follow the tone guidelines.
- Use the example structure as a template.
- Avoid using any forbidden words or phrases exactly as written.
- Be emotionally safe, accountable, and sincere.

CASE RULES DOCUMENT:
\"\"\"{rules_text}\"\"\"
    """

    user_content = f"""
Case description: {case_description}
What I did wrong: {wrongdoing}

Write a romantic, emotionally intelligent apology letter (4–8 sentences)
that follows the above rules.
"""

    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=400,
        messages=[
            {"role": "user", "content": user_content}
        ]
    )

    apology = ""
    for block in resp.content:
        if block.type == "text":
            apology += block.text

    return apology.strip()


def answer_rule_question(query: str, top_k: int = 2) -> str:
    """
    Retrieve the most relevant rules and ask Claude to explain them
    in response to the user's question. Does not generate an apology.
    """
    matches = retrieve_relevant_rules(query, k=top_k)

    if not matches:
        return (
            "I couldn't find any rules related to that question. "
            "Try rephrasing or upload the rulebook again."
        )

    combined_rules = "\n\n".join(
        f"{m['case_name']}\n{m['raw_text']}" for m in matches if m.get("raw_text")
    )

    system_prompt = (
        "You are an expert assistant who explains the 'girlfriend rulebook' clearly and gently. "
        "Answer the user's question using only the provided rules text. "
        "Be supportive, concise, and avoid inventing additional rules. "
        "When helpful, cite specific phrases or bullets from the rules."
    )

    user_content = f"""User question:
{query}

Relevant rule text (do not reveal this raw text verbatim, summarize it):
\"\"\"{combined_rules}\"\"\"

Explain the relevant rule(s) and guidance in 3–6 sentences. Avoid writing an apology; just explain the rules and tone."""

    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        system=system_prompt,
        max_tokens=400,
        messages=[{"role": "user", "content": user_content}],
    )

    answer = ""
    for block in resp.content:
        if block.type == "text":
            answer += block.text
    return answer.strip()
