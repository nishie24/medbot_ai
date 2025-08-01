import os
import zipfile
import xml.etree.ElementTree as ET
from html import unescape
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

# ---------------------------
# File paths
# ---------------------------
DATA_DIR = "data"
ENCYCLOPEDIA_PDF_PATH = os.path.join(DATA_DIR, "encyclopedia.pdf")
MEDLINEPLUS_ZIP_PATH = os.path.join(DATA_DIR, "medlineplus_health_topics.zip")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "langchain_index")

# ---------------------------
# Load MedlinePlus from ZIP
# ---------------------------
def load_medlineplus_zip(zip_path, max_chars=750, min_chars=80, limit=None):
    if not os.path.exists(zip_path):
        print("⚠ MedlinePlus ZIP not found.")
        return []

    docs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        xml_files = [f for f in z.namelist() if f.lower().endswith(".xml")]
        print(f"[INFO] Found {len(xml_files)} XML files in MedlinePlus ZIP.")

        for i, xml_file in enumerate(xml_files):
            if limit and i >= limit:
                break

            try:
                with z.open(xml_file) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    ns = {"m": root.tag.split("}")[0][1:]} if root.tag.startswith("{") else {}

                    topics = root.findall(".//m:health-topic", ns) if ns else root.findall(".//health-topic")
                    if not topics:
                        continue

                    for topic in topics:
                        def get(el_name):
                            el = topic.find(f"m:{el_name}", ns) if ns else topic.find(el_name)
                            return el.text.strip() if el is not None and el.text else ""

                        lang = get("language").lower() or "en"
                        if lang != "en":
                            continue

                        title = get("title") or "Unknown Topic"
                        summary = get("full-summary")
                        summary = unescape(summary or "").replace("\n", " ").strip()

                        if len(summary) < min_chars:
                            continue

                        text = f"Source: MedlinePlus\nTitle: {title}\n\n{summary[:max_chars]}..."
                        docs.append(Document(page_content=text, metadata={"source": "MedlinePlus"}))

            except Exception as e:
                print(f"[SKIP] Failed to parse {xml_file}: {e}")

    print(f"[INFO] Loaded {len(docs)} MedlinePlus documents.")
    return docs

# ---------------------------
# Main Build Pipeline
# ---------------------------
def main():
    docs = []

    # Load MedlinePlus
    med_docs = load_medlineplus_zip(MEDLINEPLUS_ZIP_PATH)
    docs += med_docs

    # Load Medical Encyclopedia PDF
    if os.path.exists(ENCYCLOPEDIA_PDF_PATH):
        pdf_loader = PyPDFLoader(ENCYCLOPEDIA_PDF_PATH)
        encyclopedia_docs = pdf_loader.load()
        for doc in encyclopedia_docs:
            doc.metadata["source"] = "MedicalEncyclopedia"
        docs += encyclopedia_docs

    if not docs:
        raise SystemExit("❌ No documents found. Check all data sources.")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Count chunks per source
    source_counts = Counter(doc.metadata.get("source", "Unknown") for doc in split_docs)

    # Build embeddings
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embedding_model)
    db.save_local(FAISS_INDEX_DIR)

    # Report
    print(f"\n✅ LangChain FAISS index saved to: {FAISS_INDEX_DIR}")
    print(f"📦 Total chunks: {len(split_docs)}")
    print("🔍 Chunks by source:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")

if __name__ == "__main__":
    main()
