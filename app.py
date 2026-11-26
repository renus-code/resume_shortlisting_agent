import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from graph import graph

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_text(file) -> str:
    full_text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return full_text

st.set_page_config(page_title="Agentic Resume Screener", layout="wide")
st.title("Agentic Resume Screener")

job_description = st.text_area("Paste Job Description", height=200)
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")


if st.button("Screen Resume"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Running LangGraph agent..."):
            resume_text = extract_text(uploaded_file)

            if len(resume_text) + len(job_description) > 24000:
                st.warning("This PDF is very long and may hit Groq's 8,000-token limit.")

            config = {
                "configurable": {
                    "thread_id": uploaded_file.name
                }
            }

            initial_state = {
                "resume": resume_text,
                "jd": job_description,
                "score": 0.0,
                "decision": "",
                "feedback": ""
            }

            result = graph.invoke(initial_state, config=config)

            st.write(f"Score: {result['score']}%")
            st.write(f"Decision: {result['decision']}")
            st.markdown(f"**Feedback:**\n\n{result['feedback']}")
    else:
        st.warning("Please upload a resume AND paste a job description.")
