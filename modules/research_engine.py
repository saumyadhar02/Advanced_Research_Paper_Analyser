
import json
import re

from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL


FIELDS = [
    "research_objective",
    "methodology",
    "dataset",
    "key_findings",
    "limitations",
]


def split_pages(text):
    parts = re.split(r"--- Page (\d+) ---", text)
    pages = {}

    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            page_number = int(parts[i])
            page_text = parts[i + 1].strip()
            if page_text:
                pages[page_number] = page_text
    elif text.strip():
        pages[1] = text.strip()

    return pages



def parse_json_response(content):
    import json
    import re

    content = content.strip()

    # Remove Markdown code fences if present
    content = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"\s*```\s*$", "", content)

    # First, try parsing the entire response
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find a JSON object inside any surrounding text
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", content):
        try:
            result, _ = decoder.raw_decode(content[match.start():])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "The model did not return a complete valid JSON object. "
        "Try a shorter context or a model with stronger JSON output support."
    )

def verify_evidence(evidence, pages):
    verified = []

    if not isinstance(evidence, list):
        return verified

    for item in evidence:
        if not isinstance(item, dict):
            continue

        quote = str(item.get("quote", "")).strip()

        try:
            page = int(item.get("page"))
        except (ValueError, TypeError):
            page = None

        page_text = pages.get(page, "") if page is not None else ""

        verified.append({
            "page": page if page is not None else "Unknown",
            "quote": quote,
            "verified": bool(
                quote and page_text
                and quote.casefold() in page_text.casefold()
            ),
        })

    return verified


def analyze_paper(doc):
    metadata = doc.metadata or {}
    title = metadata.get("title") or metadata.get("source") or "Untitled paper"
    source = metadata.get("source", title)

    pages = split_pages(doc.page_content)

    if not pages:
        raise ValueError(f"No readable text was extracted from {source}.")

    sections = []
    remaining = 24000

    for page_number, page_text in pages.items():
        if remaining <= 0:
            break

        excerpt = page_text[:remaining]
        sections.append(f"[PAGE {page_number}]\n{excerpt}")
        remaining -= len(excerpt)

    context = "\n\n".join(sections)

    prompt = f"""
You are a careful scientific literature analyst.
Analyze only the source text below. Do not invent facts.
If information is unavailable, write "Not reported in the provided text".

Return one valid JSON object with these fields:
- research_objective
- methodology
- dataset
- key_findings
- limitations

Each field must contain:
"value": a concise description
"evidence": a list of objects containing "page" and "quote"

Use only page numbers shown in the source labels.
Evidence quotations must be copied exactly from the supplied text.
Use up to three short quotations per field.
Return JSON only, without Markdown fences.

Paper title: {title}
Source filename: {source}

SOURCE TEXT:
{context}
"""

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    response = llm.invoke(prompt)

    result = parse_json_response(str(response.content))

    profile = {
        "title": title,
        "source": source,
        "fields": {},
    }

    for field in FIELDS:
        raw = result.get(field, {})

        if not isinstance(raw, dict):
            raw = {"value": str(raw), "evidence": []}

        profile["fields"][field] = {
            "value": str(
                raw.get("value", "Not reported in the provided text")
            ).strip(),
            "evidence": verify_evidence(raw.get("evidence", []), pages),
        }

    return profile


def analyze_corpus(documents, progress_callback=None):
    profiles = []
    errors = []

    for index, doc in enumerate(documents):
        source = (doc.metadata or {}).get(
            "source", f"Document {index + 1}"
        )

        try:
            profiles.append(analyze_paper(doc))
        except Exception as exc:
            errors.append({
                "source": source,
                "error": str(exc),
            })

        if progress_callback:
            progress_callback(index + 1, len(documents))

    return profiles, errors
