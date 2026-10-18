
import streamlit as st
import streamlit.components.v1 as components

from modules.parser import extract_papers
from modules.rag_engine import initialize_vector_store, get_rag_chain
from modules.graph_engine import (
    build_advanced_knowledge_graph,
    generate_pyvis_html,
)
from modules.research_engine import analyze_corpus


st.set_page_config(
    page_title="Advanced Research RAG & Graph Studio",
    page_icon="🔬",
    layout="wide",
)


def initialize_session_state():
    defaults = {
        "vector_store": None,
        "graph": None,
        "documents": [],
        "research_profiles": [],
        "research_errors": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_evidence(field_data):
    evidence = field_data.get("evidence", [])

    if not evidence:
        st.caption("No supporting quotation was returned.")
        return

    for item in evidence:
        page = item.get("page", "Unknown")
        quote = item.get("quote", "")
        verified = item.get("verified", False)

        status = "Quote verified in extracted text" if verified else (
            "Quote not verified — check the source manually"
        )

        st.markdown(f"**Page {page}** — {status}")

        if quote:
            st.caption(f"> {quote}")


def main():
    initialize_session_state()

    st.title("🔬 Advanced Research Paper Studio & Knowledge Graph")
    st.markdown(
        "Local research analysis powered by LangChain, Ollama, "
        "Chroma, NetworkX, and PyVis."
    )

    with st.sidebar:
        st.header("Document Ingestion")

        uploaded_files = st.file_uploader(
            "Upload Research PDFs",
            type="pdf",
            accept_multiple_files=True,
        )

        process_btn = st.button(
            "Build Knowledge Base",
            type="primary",
        )

    if process_btn:
        if not uploaded_files:
            st.sidebar.error("Please upload at least one PDF.")
        else:
            try:
                with st.spinner(
                    "Parsing papers, building embeddings and creating graph..."
                ):
                    docs = extract_papers(uploaded_files)

                    if not docs:
                        st.sidebar.error(
                            "Could not extract readable text from the PDFs."
                        )
                    else:
                        vector_store = initialize_vector_store(docs)
                        graph = build_advanced_knowledge_graph(docs)

                        st.session_state.documents = docs
                        st.session_state.vector_store = vector_store
                        st.session_state.graph = graph

                        # Profiles belong to the current uploaded corpus.
                        st.session_state.research_profiles = []
                        st.session_state.research_errors = []

                        st.sidebar.success(
                            f"Processed {len(docs)} document(s)."
                        )

            except Exception as exc:
                st.sidebar.error(f"Processing failed: {exc}")

    if st.session_state.vector_store is None:
        st.info(
            "👈 Upload your research PDFs and click "
            "**Build Knowledge Base** to begin."
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Generative Q&A",
        "🕸️ Knowledge Graph",
        "📄 Library View",
        "🧠 Research Intelligence",
    ])

    with tab1:
        st.header("Generative Literature Assistant")
        st.write(
            "Ask questions about methodologies, findings and limitations "
            "across your uploaded papers."
        )

        query = st.text_input(
            "Ask a question about your research corpus:",
            key="rag_query",
        )

        if query:
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    chain = get_rag_chain(st.session_state.vector_store)
                    response = chain.invoke({"input": query})

                    st.markdown("### Answer")
                    st.write(response.get("answer", "No answer returned."))

                    with st.expander("Retrieved source documents"):
                        for doc in response.get("context", []):
                            metadata = doc.metadata or {}
                            st.markdown(
                                f"**Source:** "
                                f"{metadata.get('source', 'Unknown')}"
                            )
                            st.write(doc.page_content[:1200])

                except Exception as exc:
                    st.error(
                        f"RAG query failed: {exc}. "
                        "Check that Ollama is running and the configured "
                        "model is available."
                    )

    with tab2:
        st.header("Interactive Knowledge Graph")
        st.write(
            "Explore the current paper–concept network. "
            "Concept connections are candidate relationships, not proof "
            "of scientific agreement."
        )

        if st.session_state.graph:
            try:
                html_data = generate_pyvis_html(
                    st.session_state.graph
                )
                components.html(html_data, height=650, scrolling=True)
            except Exception as exc:
                st.error(f"Could not render the graph: {exc}")
        else:
            st.info("The graph has not been generated.")

    with tab3:
        st.header("Uploaded Corpus")

        for doc in st.session_state.documents:
            metadata = doc.metadata or {}
            title = metadata.get("title", "Untitled document")

            with st.expander(f"📄 {title}"):
                st.write(
                    f"**File:** {metadata.get('source', 'Unknown')}"
                )
                st.text(doc.page_content[:3000])

                if len(doc.page_content) > 3000:
                    st.caption("Preview truncated.")

    with tab4:
        st.header("Research Intelligence")
        st.write(
            "Extract structured research profiles and compare what "
            "the uploaded papers report. Run this analysis using your "
            "configured local Ollama model."
        )

        st.warning(
            "Extracted information is AI-generated. Verify important "
            "claims against the original papers. A verified quotation "
            "confirms text matching, not the correctness of its interpretation."
        )

        if len(st.session_state.documents) < 2:
            st.info(
                "Upload and process at least two papers to compare "
                "their research profiles."
            )

        analyze_btn = st.button(
            "Analyze Research Papers",
            type="primary",
            disabled=not st.session_state.documents,
        )

        if analyze_btn:
            progress = st.progress(0)
            status = st.empty()

            def update_progress(current, total):
                progress.progress(current / total)
                status.write(
                    f"Analyzing paper {current} of {total}..."
                )

            with st.spinner(
                "Extracting objectives, methods, datasets, findings and limitations..."
            ):
                profiles, errors = analyze_corpus(
                    st.session_state.documents,
                    progress_callback=update_progress,
                )

                st.session_state.research_profiles = profiles
                st.session_state.research_errors = errors

            status.empty()
            progress.empty()

            if profiles:
                st.success(
                    f"Created profiles for {len(profiles)} paper(s)."
                )

            if errors:
                st.warning(
                    f"{len(errors)} paper(s) could not be analyzed."
                )

        profiles = st.session_state.research_profiles
        errors = st.session_state.research_errors

        if errors:
            with st.expander("Analysis errors"):
                for error in errors:
                    st.error(
                        f"{error['source']}: {error['error']}"
                    )

        if profiles:
            st.subheader("Cross-Paper Comparison")

            field_labels = {
                "research_objective": "Research objective",
                "methodology": "Methodology",
                "dataset": "Dataset",
                "key_findings": "Key findings",
                "limitations": "Limitations",
            }

            comparison_rows = []

            for profile in profiles:
                row = {"Paper": profile["title"]}

                for field, label in field_labels.items():
                    row[label] = profile["fields"][field]["value"]

                comparison_rows.append(row)

            st.dataframe(
                comparison_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Paper Profiles and Evidence")

            for profile in profiles:
                with st.expander(f"📑 {profile['title']}"):
                    st.caption(f"Source file: {profile['source']}")

                    for field, label in field_labels.items():
                        field_data = profile["fields"][field]

                        st.markdown(f"**{label}**")
                        st.write(field_data["value"])

                        with st.expander(f"Evidence for {label}"):
                            render_evidence(field_data)

            st.caption(
                "This initial version compares extracted profiles. "
                "It does not yet automatically establish contradictions "
                "or certify research gaps."
            )


if __name__ == "__main__":
    main()
