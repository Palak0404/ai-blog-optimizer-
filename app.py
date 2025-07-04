import streamlit as st  # type: ignore
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import google.generativeai as genai  # type: ignore

genai.configure(api_key=st.secrets["gcp"]["GEMINI_API_KEY"])

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

def fetch_blog_content(url):  # scraping
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"Error: {e}"

def generate_multiple_metadata_with_gemini(content):  # Gemini: 3 titles + 3 descriptions
    prompt = (
      f"From the blog content below, generate 3 different SEO-optimized Page Titles (each max 70 characters) "
        f"and 3 different SEO Meta Descriptions (each max 160 characters).\n\n"
        f"Blog Content:\n{content}\n\n"
        f"Format:\n"
        f"Title 1: ...\nDescription 1: ...\n"
        f"Title 2: ...\nDescription 2: ...\n"
        f"Title 3: ...\nDescription 3: ..."
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# Streamlit UI
st.set_page_config(page_title="AI Blog Meta Generator")
st.title("AI Blog Meta Generator")

url = st.text_input("Enter a Blog URL:")

if url and st.button("Generate with Gemini"):
    with st.spinner("Fetching blog content..."):
        content = fetch_blog_content(url)

    if "Error:" in content:
        st.error(content)
    else:
        st.markdown("### Gemini Output (3 Titles + 3 Descriptions)")
        result = generate_multiple_metadata_with_gemini(content)
        st.text(result)
