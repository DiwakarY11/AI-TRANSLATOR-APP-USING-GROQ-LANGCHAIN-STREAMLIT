import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Retrieve API Key from environment or Streamlit Cloud Secrets
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

# ===== UI Layout =====
st.set_page_config(page_title="AI Translator App", page_icon="🌐", layout="centered")

st.title("🌐 AI Translator App")
st.markdown("Translate text into multiple languages seamlessly using Groq & LangChain.")
st.divider()

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Groq Model:",
        options=[
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "groq/compound-mini",
            "openai/gpt-oss-20b"
        ],
        index=0,
        help="Choose an active Groq LLM model."
    )
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")

# UI Controls
language_to_translate = st.selectbox(
    label="Language to translate to:",
    options=["English", "Spanish", "French", "Japanese", "German", "Hindi", "Italian", "Portuguese", "Russian", "Chinese"] 
)

text_to_translate = st.text_area("Paste text here:", height=150, placeholder="Enter text to translate...")

translate_btn = st.button("Translate", type="primary")

# Prompt Template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a professional translator. Your task is to accurately translate the provided text to {language}. Return only the translation."), 
        ("user", "{text}")
    ]
)

# Stream logic
if translate_btn:
    if not groq_api_key:
        st.error("⚠️ Groq API Key is missing! Please configure `GROQ_API_KEY` in environment variables / Streamlit Secrets, or enter it in the sidebar.")
    elif text_to_translate.strip() != "":
        try:
            groq_llm = ChatGroq(
                model=model_choice,
                groq_api_key=groq_api_key
            )
            
            prompt = chat_prompt_template.invoke({
                "language": language_to_translate,
                "text": text_to_translate
            })

            placeholder = st.empty()
            full_translation = ""
            
            for chunk in groq_llm.stream(prompt):
                full_translation += chunk.content
                placeholder.markdown(full_translation)
                
            st.balloons()
            
        except Exception as e:
            st.error(f"Translation Error: {str(e)}")
    else:
        st.warning("Please enter some text before clicking Translate.")