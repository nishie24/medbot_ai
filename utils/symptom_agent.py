import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.symptom_checker import predict_diseases, format_symptom_response

# -------------------
# Load environment variables
# -------------------
load_dotenv()

# -------------------
# Initialize Gemini LLM
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

# -------------------
# Store chat history
# -------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------
# Define LangChain Tool
# -------------------
def symptom_tool() -> Tool:
    return Tool(
        name="SymptomChecker",
        func=lambda symptoms: format_symptom_response(symptoms, predict_diseases(symptoms, top_n=3)),
        description="Use this tool to predict possible diseases based on a comma-separated list of symptoms. "
                    "Example: 'fever, cough, headache'"
    )

# -------------------
# Initialize LangChain Agent
# -------------------
tools = [symptom_tool()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)
