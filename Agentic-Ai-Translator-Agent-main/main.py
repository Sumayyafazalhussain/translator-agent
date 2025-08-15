import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set in the .env file.")
    st.stop()

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translator = Agent(
    name='Translator Agent',
    instructions="""
    You are a translator agent. Your job is to translate any given text from one language to another 
    as per the user's request. Maintain the correct grammar, tone, and meaning of the original text. 
    Only return the translated result without explanation or extra output.
    """
)

async def translate_async(prompt):
    response = await Runner.run(translator, input=prompt, run_config=config)
    return response.final_output

# Page config and styling
st.set_page_config(page_title="🌍 Gemini Translator", layout="centered")

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .title {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stTextArea>div>textarea {
        font-size: 1rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1.5px solid #ddd;
        transition: border-color 0.3s ease;
    }
    .stTextArea>div>textarea:focus {
        border-color: #4caf50;
        outline: none;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1.5px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🌍 AI Translator Agent </h1>', unsafe_allow_html=True)

language_options = [
    "Spanish", "French", "Urdu", "Arabic", "German", "Chinese", "Hindi", "Turkish", "Italian"
]
target_lang = st.selectbox("Translate to:", language_options, help="Select the language you want to translate to.")

input_text = st.text_area("Enter text to translate", height=180, placeholder="Type or paste text here...")

translate_btn = st.button("Translate")

if translate_btn:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("⏳ Translating..."):
            prompt = f'Translate to {target_lang}: "{input_text}"'
            translated_text = asyncio.run(translate_async(prompt))
        st.success("✅ Translation completed!")
        st.markdown(f"### Translated text in {target_lang}:")
        st.write(translated_text)

st.markdown("</div>", unsafe_allow_html=True)
