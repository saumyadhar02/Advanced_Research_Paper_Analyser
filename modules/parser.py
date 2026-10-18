from pypdf import PdfReader
from langchain_core.documents import Document

def extract_papers(uploaded_files):
    documents = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n--- Page {i+1} ---\n" + text
            
            if full_text.strip():
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": file.name,
                        "title": file.name.replace(".pdf", "").replace("_", " ")
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    return documents