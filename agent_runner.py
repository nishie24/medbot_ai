import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from symptom_agent import symptom_tool

# -------------------
# Load API key from .env
# -------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# -------------------
# Initialize Gemini LLM
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.2,
)

# -------------------
# Set up tools and agent
# -------------------
tools = [symptom_tool()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# -------------------
# Public runner function
# -------------------
def run_symptom_agent(user_input: str) -> str:
    try:
        return agent.run(user_input)
    except Exception as e:
        return f"❌ Agent Error: {e}"
