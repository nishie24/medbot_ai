import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from utils.rag_retriever import retrieve_context
from prompt_templates import build_rag_prompt
from utils.symptom_checker import predict_diseases, format_symptom_response

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
DEFAULT_MAX_NEW_TOKENS = 2048

# -------------------
# UI Config
# -------------------
st.set_page_config(
    page_title="MedBot AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS
# -------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(75,108,183,0.4);
    }
    .result-card {
        background: white;
        color: #222;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-left: 5px solid #4b6cb7;
    }
    .symptom-chip {
        display: inline-block;
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        color: #1976d2;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# Load Symptoms
# -------------------
@st.cache_data
def load_all_symptoms():
    try:
        df = pd.read_csv("data/disease_symptom.csv")
        if "Disease" not in df.columns:
            st.error("CSV missing 'Disease' column.")
            return []
        symptoms = df.columns.tolist()
        symptoms.remove("Disease")
        return sorted([s.strip().lower().replace(" ", "_").replace("-", "_") for s in symptoms])
    except Exception as e:
        st.error(f"Error loading symptoms: {e}")
        return []

ALL_SYMPTOMS = load_all_symptoms()

# -------------------
# Session State
# -------------------
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "🧠 Medical Q&A"
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []

# -------------------
# Header
# -------------------
st.markdown("""
<div class="header">
    <h1>🧬 MedBot AI </h1>
    <p>AI-Driven Medical Intelligence</p>
</div>
""", unsafe_allow_html=True)

# -------------------
# Mode Switch
# -------------------
st.markdown("### Select Consultation Mode")
col1, col2 = st.columns(2)
with col1:
    if st.button("🧠 Medical Q&A", use_container_width=True):
        st.session_state.current_mode = "🧠 Medical Q&A"
with col2:
    if st.button("🩺 Symptom Diagnosis", use_container_width=True):
        st.session_state.current_mode = "🩺 Symptom Diagnosis"

mode = st.session_state.current_mode

# -------------------
# Input Section
# -------------------
st.markdown("---")
if mode == "🩺 Symptom Diagnosis":
    st.markdown("### 📋 Select Your Symptoms")
    selected = st.multiselect(
        "Search symptoms:",
        options=ALL_SYMPTOMS,
        default=st.session_state.selected_symptoms,
        format_func=lambda x: x.replace("_", " ").title(),
        key="symptom_selector",
        label_visibility="collapsed"
    )
    st.session_state.selected_symptoms = selected
    user_input = ", ".join(selected) if selected else ""
else:
    st.markdown("### 💬 Ask Your Medical Question")
    user_input = st.text_area(
        "Enter your question:",
        placeholder="e.g. What are symptoms of COVID-19?",
        height=150,
        label_visibility="collapsed"
    )

# -------------------
# Gemini Response Generator
# -------------------
def generate_gemini_response(question: str):
    try:
        context_str, context_items = retrieve_context(question)
        prompt = build_rag_prompt(question, context_str)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": DEFAULT_MAX_NEW_TOKENS, "temperature": 0.2}
        )
        if not response.candidates or not response.candidates[0].content.parts:
            return ("⚠ No response returned.", prompt, context_items)
        return (response.candidates[0].content.parts[0].text.strip(), prompt, context_items)
    except Exception as e:
        return (f"❌ Error: {e}", "", [])

# -------------------
# Run Button & Output
# -------------------
if st.button("🔍 Analyze", type="primary"):
    if not user_input.strip():
        st.warning("⚠ Please enter input before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            if mode == "🧠 Medical Q&A":
                answer, _, chunks_used = generate_gemini_response(user_input)
                st.markdown(f"""
                    <div class="result-card">
                        <h3>💡 AI Response</h3>
                        <hr>
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                if chunks_used:
                    with st.expander("📚 Reference Context"):
                        for i, (source, chunk) in enumerate(chunks_used, 1):
                            st.markdown(f"**{i}.** `{source}`")
                            st.code(chunk[:350] + ("..." if len(chunk) > 350 else ""))
            else:
                current_symptoms = st.session_state.selected_symptoms
                if not current_symptoms:
                    st.warning("⚠ Please select symptoms.")
                else:
                    if len(current_symptoms) == 1:
                        st.info("ℹ️ Diagnosing with a single symptom may produce broader or less accurate results.")
                    result_df = predict_diseases(user_input, top_n=3, min_score=0.0)
                    formatted = format_symptom_response(user_input, result_df)
                    chips_html = ' '.join([
                        f'<span class="symptom-chip">{s.replace("_", " ").title()}</span>'
                        for s in current_symptoms
                    ])
                    st.markdown(f"""
                        <div class="result-card">
                            <h3>🩺 Diagnostic Results</h3>
                            <p><strong>Selected Symptoms:</strong></p>
                            {chips_html}
                            <hr>
                            {formatted}
                        </div>
                    """, unsafe_allow_html=True)


# -------------------
# Footer
# -------------------
st.markdown("""
<div class="footer">
    <p> For Educational and Academic use only | Consult medical professionals for clinical decisions</p>
</div>
""", unsafe_allow_html=True)
