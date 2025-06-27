import streamlit as st  # type: ignore
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import google.generativeai as genai  # type: ignore
from transformers import pipeline  # type: ignore

# Gemini API Key from Streamlit Secrets
genai.configure(api_key=st.secrets["gcp"]["GEMINI_API_KEY"])

# Load T5 Summarizer
t5_summarizer = pipeline("summarization", model="t5-small")

# HTTP headers for scraping
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

# Function to scrape blog content
def fetch_blog_content(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"Error: {e}"

# Gemini Metadata Generator
def generate_metadata_with_gemini(content):
    prompt = (
        f"From the following blog content, generate:\n\n"
        f"1. A catchy SEO Page Title (max 70 characters)\n"
        f"2. An SEO Meta Description (max 160 characters)\n\n"
        f"Blog Content:\n{content}\n\n"
        f"Format:\nTitle: ...\nDescription: ..."
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# T5 Metadata Generator
def generate_metadata_with_t5(content):                                             # T5-Small
    try:
        summary = t5_summarizer(content[:1024],max_length=320,min_length=150,do_sample=True,early_stopping=False,no_repeat_ngram_size=2)[0]['summary_text']
        sentences = [s.strip() for s in summary.strip().split('.') if s.strip()]
        title = next((s for s in sentences if len(s) <= 70 and len(s) > 20), None)
        if not title and sentences:
            fallback = sentences[0][:70]
            title = ' '.join(fallback.split(' ')[:-1]) 
        description = ''
        for s in sentences:
            if s == title:
                continue
            temp = f"{description} {s}".strip()
            if len(temp) <= 160:
                description = temp
            else:
                break
        if title and not title.endswith('.'):
            title += '.'
        if description and not description.endswith('.'):
            description += '.'
        return title.strip(), description.strip()
    except Exception as e:
        return f"T5 error: {e}", ""

# Streamlit UI
st.set_page_config(page_title="AI Blog Optimizer")
st.title("AI Blog Optimizer")

url = st.text_input("Enter a Shopify blog URL:")
model_option = st.selectbox("Choose a model:", ["Gemini", "T5-Small", "Compare Both"])

if url and st.button("Generate Metadata"):
    with st.spinner("Fetching blog content..."):
        content = fetch_blog_content(url)

    if "Error:" in content:
        st.error(content)
    else:
        if model_option in ["Gemini", "Compare Both"]:
            st.markdown("## Gemini Output")
            gemini_text = generate_metadata_with_gemini(content)
            st.markdown(gemini_text)

        if model_option in ["T5-Small", "Compare Both"]:
            st.markdown("## T5-Small Output")
            title, desc = generate_metadata_with_t5(content)
            st.markdown(f"**Title ({len(title)}/70):** {title}")
            st.markdown(f"**Description ({len(desc)}/160):** {desc}")
