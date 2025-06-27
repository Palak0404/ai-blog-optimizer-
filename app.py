import streamlit as st  # type: ignore
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import google.generativeai as genai  # type: ignore
from transformers import pipeline  # type: ignore
import torch # for deployment
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

t5_summarizer = pipeline("summarization", model="t5-small")
bart_summarizer = pipeline("summarization", model="facebook/bart-base")

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

def fetch_blog_content(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"Error: {e}"

def generate_metadata_with_gemini(content):
    prompt = (
        f"From the following blog content, generate:\n\n"
        f"1. A catchy SEO Page Title (min 70 characters)\n"
        f"2. An SEO Meta Description (min 160 characters)\n\n"
        f"Blog Content:\n{content}\n\n"
        f"Format:\nTitle: ...\nDescription: ..."
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        title_line = next((line for line in response.splitlines() if line.lower().startswith("title:")), "")
        desc_line = next((line for line in response.splitlines() if line.lower().startswith("description:")), "")
        title = title_line.replace("Title:", "").strip()[:70]
        description = desc_line.replace("Description:", "").strip()[:160]

        return f"**Title:** {title}\n\n**Description:** {description}"
    except Exception as e:
        return f"Gemini error: {e}"

def generate_metadata_with_t5(content):
    try:
        summary = t5_summarizer(
            content[:1024],
            max_length=256,
            min_length=100,
            do_sample=True,
            early_stopping=True,
            no_repeat_ngram_size=2
        )[0]['summary_text']

        sentences = summary.strip().split('. ')
        title = sentences[0].strip() if sentences else summary.strip()
        description = '. '.join(sentences[1:]).strip() if len(sentences) > 1 else summary.strip()

        # Apply character limits
        return title[:70], description[:160]
    except Exception as e:
        return f"T5 error: {e}", ""


def generate_metadata_with_bart(content):
    try:
        summary = bart_summarizer(
            content[:1024],
            max_length=350,
            min_length=180,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            early_stopping=False,
            no_repeat_ngram_size=2
        )[0]['summary_text']

        sentences = summary.strip().split('. ')
        title = sentences[0].strip() if sentences else summary.strip()
        description = '. '.join(sentences[1:]).strip() if len(sentences) > 1 else summary.strip()

        # Apply character limits
        return title[:70], description[:160]
    except Exception as e:
        return f"BART error: {e}", ""


st.set_page_config(page_title="AI Blog Optimizer")                                                   #streamlit ui
st.title(" AI Blog Optimizer")
st.markdown("Generate SEO-optimized metadata from your Shopify blog using AI models.")

url = st.text_input("Enter a Shopify blog URL:")
model_option = st.selectbox("Choose a model:", ["Gemini", "T5-Small", "BART-Base", "Compare All"])

if url and st.button("Generate Metadata"):
    with st.spinner("Fetching blog content..."):
        content = fetch_blog_content(url)

    if "Error:" in content:
        st.error(content)
    else:
        if model_option in ["Gemini", "Compare All"]:
            st.markdown("##  Gemini Output")
            gemini_output = generate_metadata_with_gemini(content)
            st.markdown(gemini_output)

        if model_option in ["T5-Small", "Compare All"]:
            st.markdown("##  T5-Small Output")
            title, desc = generate_metadata_with_t5(content)
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Description:** {desc}")
            st.markdown(f"**Title ({len(title)}/70):** {title}")
            st.markdown(f"**Description ({len(desc)}/160):** {desc}")

        if model_option in ["BART-Base", "Compare All"]:
            st.markdown("## BART-Base Output")
            title, desc = generate_metadata_with_bart(content)
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Description:** {desc}")
            st.markdown(f"**Title ({len(title)}/70):** {title}")
            st.markdown(f"**Description ({len(desc)}/160):** {desc}")
