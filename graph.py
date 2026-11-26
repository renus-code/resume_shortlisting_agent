from typing import TypedDict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from groq import APIStatusError
import streamlit as st

class ResumeState(TypedDict):
    resume:str
    jd:str
    score:float
    decision:str
    feedback: str

LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, batch_size=16)

def score_resume(state: ResumeState):
    embeddings = get_embeddings()
    resume_vector = embeddings.embed_documents([state["resume"]])
    jd_vector = embeddings.embed_query(state["jd"])
    raw_score = cosine_similarity(resume_vector, [jd_vector])[0][0]
    state["score"] = round(float(raw_score) * 100, 2 )

    return state

def make_decision(state:ResumeState):
    score = state["score"]
    if score >= 70:
        state["decision"] = "shortlist"
    elif score >= 50:
        state["decision"] = "Human Review"
    else:
        state["decision"] = "reject"
    
    return state

def route_decision(state: ResumeState):
    return state["decision"]


def get_llm():
    return ChatGroq(model = LLM_MODEL, temperature=0.1)


def safe_llm_invoke(prompt: str) -> str:
    try:
        return get_llm().invoke(prompt).content
    except APIStatusError as e:
        if e.status_code == 413:
            return (
                "Uploaded Resume or Job description are too large"
                "Please upload a shorter resume (1-3 pages) or shorten the job description."
            )
        return f"**Groq API Error ({e.status_code}):** {e.message}"
    except Exception as e:
        return f"**Unexpected Error:** {str(e)}"


def shortlist(state:ResumeState):
    prompt = f"""
    System prompt: ACt as the senior Recruiter

    This candidate scored {state["score"]}% match and is shortlisted
    1. List their top 3 matching strengths for this jd
    2. Suggest 5 tailored i nterview questions

    JD: {state["jd"]}
    Resume: {state["resume"]}
    
    """
    
    state["feedback"] = safe_llm_invoke(prompt=prompt)

    return state

def human_review(state: ResumeState):
    prompt = f"""
    Act as a senior recruiter. This candidate scored {state["score"]}% match and needs HUMAN REVIEW.
    1. Summarize what aligns with the JD.
    2. List the unclear or missing requirements.
    3. Suggest 2 quick phone-screen questions to verify their fit.

    JD: {state["jd"]}
    Resume: {state["resume"]}
    """
    state["feedback"] = safe_llm_invoke(prompt=prompt)

    return state 


def reject(state: ResumeState):
    prompt = f"""
    Act as a senior recruiter. This candidate scored {state["score"]}% match and is REJECTED.
    Provide 3 concise bullet points explaining the main skill or experience gaps compared to the JD.

    JD: {state["jd"]}
    Resume: {state["resume"]}
    """
    state["feedback"] = safe_llm_invoke(prompt=prompt)

    return state 


builder = StateGraph(ResumeState)

builder.add_node("score_resume", score_resume)
builder.add_node("make_decision", make_decision)
builder.add_node("shortlist", shortlist)
builder.add_node("human_review", human_review)
builder.add_node("reject", reject)

builder.add_edge(START, "score_resume")
builder.add_edge("score_resume", "make_decision")

builder.add_conditional_edges(
    "make_decision",
    route_decision,
    {
        "shortlist":"shortlist",
        "Human Review" : "human_review",
        "reject":"reject"
    },
)

builder.add_edge("shortlist", END)
builder.add_edge("human_review", END)
builder.add_edge("reject", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)






