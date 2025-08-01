# 🧬 MedBot AI

**MedBot AI** is an AI-powered medical assistant that offers:
- 🩺 **Symptom Diagnosis** based on selected symptoms
- 🧠 **Medical Q&A** using RAG (Retrieval-Augmented Generation) with Gemini 
- 🔍 Built with LangChain, FAISS, and Gemini 1.5 Flash
- 📚 Backed by medical data from MedlinePlus and a general medical encyclopedia

---

## 🚀 Features

### 1. Symptom Diagnosis
- Select one or more symptoms from a dynamic list
- Predicts top 3 most probable diseases using a custom rule-based matcher (Jaccard, coverage, precision weighted)
- Enhanced with Gemini Agent fallback (LangChain agent)

### 2. Medical Q&A
- Ask natural language medical questions
- Uses Gemini 1.5 Flash + LangChain RAG
- Retrieves context from medical datasets (MedlinePlus + PDF encyclopedia)
- Shows AI response and reference documents used

---

## 📁 Project Structure

```
medbot_ai/
│
├── app.py                      # Streamlit frontend for MedBot AI
├── agent_runner.py            # LangChain agent for symptom diagnosis
├── build_langchain_kb.py      # Index builder (MedlinePlus XML + PDF)
├── requirements.txt           # Python dependencies
│
├── data/
│   └── disease_symptom.csv    # Rule-based symptoms dataset
│
├── prompts/
│   └── prompt_templates.py    # Gemini RAG prompt templates
│
├── utils/
│   ├── rag_retriever.py       # LangChain retriever setup (FAISS + embeddings)
│   ├── symptom_agent.py       # Symptom agent using LangChain Tool
│   ├── symptom_checker.py     # Rule-based prediction logic
│   └── text_utils.py          # Utility for text preprocessing
```

---

## 🔧 Setup Instructions

1. **Install requirements**
```bash
pip install -r requirements.txt
```

2. **Add your Gemini API key**
```bash
# .env file
GEMINI_API_KEY=your_key_here
```

3. **Prepare knowledge base**
```bash
python build_langchain_kb.py
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## 🧠 Tech Stack

| Component | Technology |
|----------|-------------|
| UI       | Streamlit |
| LLM      | Gemini 1.5 Flash (Google Generative AI) |
| RAG      | LangChain + FAISS + SentenceTransformers |
| Agent    | LangChain Tool + Structured Chat |
| Embedding Model | all-MiniLM-L6-v2 |
| Data     | MedlinePlus XML + Encyclopedia PDF + Symptom CSV |



## ⚠ Disclaimer

This tool is for **educational and academic use only**.
Always consult certified healthcare professionals for medical concerns.