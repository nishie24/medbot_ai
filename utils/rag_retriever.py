from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------
# Paths & Config
# -------------------
INDEX_DIR = "data/langchain_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# -------------------
# LangChain Retriever
# -------------------
def get_langchain_retriever(top_k: int = 3):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
    INDEX_DIR,
    embedding_model,
    allow_dangerous_deserialization=True  # ✅ Safe if index is yours
)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever

# -------------------
# Retrieve Function
# -------------------
def retrieve_context(query: str, top_k: int = 3, max_context_chars: int = 1200):
    retriever = get_langchain_retriever(top_k)
    docs = retriever.invoke(query)

    selected_chunks = []
    context_items = []
    total_len = 0

    for doc in docs:
        chunk = doc.page_content
        c_len = len(chunk)
        if total_len + c_len > max_context_chars:
            chunk = chunk[: max_context_chars - total_len] + "..."
        selected_chunks.append(chunk)
        context_items.append((doc.metadata.get("source", "Unknown"), chunk))
        total_len += len(chunk)
        if total_len >= max_context_chars:
            break

    context_str = "\n\n---\n\n".join(selected_chunks)
    return context_str, context_items  # ✅ return both
