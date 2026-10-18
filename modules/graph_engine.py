
import re
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
from pyvis.network import Network


STOP_WORDS = {
    "about", "after", "again", "also", "among", "because",
    "been", "before", "being", "between", "both", "could",
    "does", "during", "each", "from", "have", "having",
    "into", "more", "most", "other", "over", "same",
    "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through",
    "under", "very", "were", "what", "when", "where",
    "which", "while", "with", "would", "your", "using",
    "used", "based", "paper", "papers", "study", "result",
    "results", "method", "methods", "approach", "model",
    "models", "data", "figure", "table", "section",
    "introduction", "conclusion", "university", "research",
    "author", "authors", "however", "therefore", "thus",
    "since", "within", "without", "given", "many", "some",
    "will", "can", "may", "one", "two", "three", "http",
    "www", "arxiv", "preprint",
}


def extract_page_evidence(text):
    """Split extracted PDF text using parser.py page markers."""
    pages = {}
    parts = re.split(r"--- Page (\d+) ---", text)

    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            page_number = int(parts[i])
            page_text = parts[i + 1].strip()
            pages[page_number] = page_text
    else:
        pages[1] = text

    return pages


def extract_concepts(text, max_concepts=12):
    """
    Extract frequent words and adjacent two-word phrases.
    These are candidate concepts, not verified scientific entities.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())

    cleaned = [
        word for word in words
        if word not in STOP_WORDS and len(word) >= 4
    ]

    counts = Counter(cleaned)

    bigrams = Counter(
        f"{left} {right}"
        for left, right in zip(cleaned, cleaned[1:])
        if left not in STOP_WORDS
        and right not in STOP_WORDS
        and left != right
    )

    candidates = []

    for phrase, frequency in bigrams.items():
        if frequency >= 2:
            candidates.append((phrase, frequency * 1.5))

    for word, frequency in counts.items():
        if frequency >= 2:
            candidates.append((word, float(frequency)))

    candidates.sort(key=lambda item: (-item[1], item[0]))

    selected = []
    seen = set()

    for concept, score in candidates:
        if concept not in seen:
            selected.append((concept, score))
            seen.add(concept)

        if len(selected) >= max_concepts:
            break

    return selected


def build_advanced_knowledge_graph(documents):
    G = nx.Graph()

    concept_documents = defaultdict(set)
    concept_pages = defaultdict(lambda: defaultdict(set))

    for index, doc in enumerate(documents):
        metadata = doc.metadata or {}

        title = (
            metadata.get("title")
            or metadata.get("source")
            or f"Document {index + 1}"
        )
        source = metadata.get("source", title)
        paper_id = f"paper::{index}::{source}"

        pages = extract_page_evidence(doc.page_content)

        G.add_node(
            paper_id,
            label=title,
            title=f"Paper: {title}\nSource: {source}",
            color="#6366f1",
            size=32,
            group="Papers",
            node_type="paper",
            source=source,
        )

        paper_text = " ".join(pages.values())
        concepts = extract_concepts(paper_text)

        for concept, score in concepts:
            concept_id = f"concept::{concept}"

            matched_pages = [
                page_number
                for page_number, page_text in pages.items()
                if re.search(
                    rf"(?<!\w){re.escape(concept)}(?!\w)",
                    page_text,
                    re.IGNORECASE,
                )
            ]

            concept_documents[concept].add(paper_id)
            concept_pages[concept][paper_id].update(matched_pages)

            if not G.has_node(concept_id):
                G.add_node(
                    concept_id,
                    label=concept,
                    title=f"Candidate concept: {concept}",
                    color="#10b981",
                    size=14,
                    group="Concepts",
                    node_type="concept",
                )

            G.add_edge(
                paper_id,
                concept_id,
                title="Mentions candidate concept",
                label="mentions",
                color="#64748b",
                width=1,
                relationship="mentions",
            )

    # Connect papers that share at least two candidate concepts.
    paper_pairs = defaultdict(list)

    for concept, paper_ids in concept_documents.items():
        for paper_a, paper_b in combinations(sorted(paper_ids), 2):
            paper_pairs[(paper_a, paper_b)].append(concept)

    for (paper_a, paper_b), shared in paper_pairs.items():
        if len(shared) >= 2:
            shared_labels = ", ".join(sorted(shared)[:8])

            G.add_edge(
                paper_a,
                paper_b,
                title=f"Shared candidate concepts: {shared_labels}",
                label=f"{len(shared)} shared",
                color="#f59e0b",
                width=min(1 + len(shared), 5),
                dashes=True,
                relationship="shares_concepts",
            )

    # Attach source and page evidence to each concept node.
    for concept, paper_ids in concept_documents.items():
        concept_id = f"concept::{concept}"
        evidence_lines = []

        for paper_id in sorted(paper_ids):
            source = G.nodes[paper_id]["source"]
            page_numbers = sorted(
                concept_pages[concept][paper_id]
            )

            if page_numbers:
                evidence_lines.append(
                    f"{source}: page(s) "
                    + ", ".join(map(str, page_numbers[:15]))
                )

        G.nodes[concept_id]["title"] = (
            f"Candidate concept: {concept}\n"
            f"Found in {len(paper_ids)} document(s)\n"
            + "\n".join(evidence_lines)
        )
        G.nodes[concept_id]["document_count"] = len(paper_ids)

    return G


def generate_pyvis_html(G, output_path="graph.html"):
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )

    net.from_nx(G)

    net.barnes_hut(
        gravity=-2500,
        central_gravity=0.25,
        spring_length=150,
        spring_strength=0.03,
        damping=0.09,
    )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "nodes": {
        "font": {
          "size": 16,
          "face": "Arial"
        },
        "borderWidth": 1
      },
      "edges": {
        "smooth": {
          "type": "dynamic"
        },
        "font": {
          "size": 10,
          "color": "#d1d5db",
          "strokeWidth": 0
        }
      },
      "physics": {
        "stabilization": {
          "iterations": 180
        }
      }
    }
    """)

    # Generate HTML in memory, avoiding PyVis save_graph's
    # default Windows encoding.
    html_data = net.generate_html()

    # Explicitly write the generated HTML as UTF-8.
    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.write(html_data)

    return html_data
