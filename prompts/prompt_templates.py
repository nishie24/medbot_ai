def build_rag_prompt(question: str, context: str) -> str:
    """
    Builds the prompt for RAG-based medical Q&A.
    Automatically adds 'When to see a doctor' guidance if question/context implies it.
    """

    # Simple heuristics
    q_lower = question.lower()
    c_lower = context.lower()

    should_mention_doctor = any(kw in q_lower for kw in [
        "when should", "should i see", "see a doctor", "urgent", "emergency", "consult"
    ]) or "see a doctor" in c_lower or "consult" in c_lower or "emergency" in c_lower

    # Doctor section toggle
    doctor_section = (
        "5. If the user's question includes or implies concern about severity "
        "(e.g., 'should I see a doctor'), or the context provides relevant advice, "
        "include a final section titled **'When to See a Doctor'** summarizing public guidance.\n"
        if should_mention_doctor else
        "5. Only include a **'When to See a Doctor'** section if the context contains such advice or the user explicitly asks about it.\n"
    )

    return (
        "You are a helpful and trusted medical assistant trained to support academic learning and public health education. "
        "This response is for informational purposes only—not for diagnosis, treatment, or medical advice.\n\n"
        "You will receive a **medical question** and **retrieved knowledge base context** from trusted sources "
        "(MedlinePlus, Gale Encyclopedia, Kaggle disease datasets).\n\n"
        "Your task:\n"
        "1. Read and understand the user's question carefully.\n"
        "2. Analyze the provided context and use only that content to answer accurately.\n"
        "3. Be clear, factual, and educational. Avoid speculation.\n"
        "4. Structure your response using bullet points or short sections when useful.\n"
        f"{doctor_section}"
        "6. If the context is insufficient or unclear, say so politely and avoid guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{question}\n\n"
        "Now write a clear, structured, and educational response:\n\nAnswer:"
    )


def build_symptom_prompt(user_symptoms: str, context_str: str) -> str:
    """
    Builds a prompt for symptom-based prediction using retrieved knowledge. 
    Optimized for Gemini 1.5 Flash to give a helpful and clear explanation.
    """

    return (
        "You are a responsible AI assistant trained to help users understand possible medical conditions for learning and awareness. "
        "Your response is for academic and informational use only—not a diagnosis or medical advice.\n\n"

        "You will receive:\n"
        "- A list of symptoms provided by the user\n"
        "- A set of relevant medical context entries from trusted sources (MedlinePlus, Gale Encyclopedia)\n\n"

        "Your tasks:\n"
        "1. Analyze the user's symptoms and compare them with the context.\n"
        "2. Identify possible matching conditions based on context (not from your own knowledge).\n"
        "3. Explain your reasoning clearly and use structured formatting (bullets, sections).\n"
        "4. If symptoms are vague or generic, state that clearly.\n"
        "5. If context mentions red flags or when to consult a doctor, include a final section titled **'When to See a Doctor'**.\n\n"

        f"User Symptoms:\n{user_symptoms}\n\n"
        f"Relevant Context:\n{context_str}\n\n"
        "Now provide a clear, factual, and educational explanation:\n\nAnswer:"
    )
