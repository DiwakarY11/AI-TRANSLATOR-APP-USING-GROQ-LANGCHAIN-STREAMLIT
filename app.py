import streamlit as st
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI # Commented out to avoid error
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# ===== UI ===========
st.title("AI Translator App")
st.divider()
st.markdown("## Translate text to any language.")

# UI Select-box - user will pick the language to translate to
language_to_translate = st.selectbox(
    label="Language to translate to:",
    options=["English", "Spanish", "French", "Japanese", "German", "Hindi"] 
)

# UI Text Area - user will paste the text in here
text_to_translate = st.text_area("Paste text here:")

# UI Button - user will press the button when they're ready to translate the text
translate_btn = st.button("Translate")


# Models: Updated to a currently supported Groq model
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
# openai_llm = ChatOpenAI() # Commented out because you don't have an OpenAI key

# Chat Prompt Template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a professional translator. You task is to translate the following text to {language}"), 
        ("user", "{text}")
    ]
)

# If the button is pressed and the text area is not empty then we start translating.
if translate_btn and text_to_translate.strip() != "":
    
    # Prompt logic moved inside the button block
    prompt = chat_prompt_template.invoke({
        "language": language_to_translate,
        "text": text_to_translate
    })

    placeholder = st.empty() # empty placeholder for the translation
    full_translation = "" # string to concatenate the chunks of text from the model
    
    # we stream the response from the model
    for chunk in groq_llm.stream(prompt):
        full_translation += chunk.content # concat each chunk
        placeholder.markdown(full_translation) # show concatenated string in the UI
        
    st.balloons() # <- balloons just for fun!

elif translate_btn:
    st.warning("Please enter some text before clicking Translate.")