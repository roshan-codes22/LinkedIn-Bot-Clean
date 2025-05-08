import os
import json
import torch
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="LinkedIn Podcast Bot", layout="centered")
torch._classes = {}  # Fix for Streamlit + torch

# --- Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- Model ---
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Load and split documents ---
@st.cache_data
def load_documents():
    import glob
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    for filepath in glob.glob("transcripts/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            episode = os.path.basename(filepath).replace(".txt", "")
            splits = splitter.split_text(text)
            for chunk in splits:
                documents.append(Document(page_content=chunk, metadata={"episode": episode}))
    return documents

# --- Load vectorstore ---
@st.cache_resource
def get_vectorstore(_documents):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding)
    db.save_local("transcript_faiss_index")
    return FAISS.load_local("transcript_faiss_index", embedding, allow_dangerous_deserialization=True)

documents = load_documents()
db = get_vectorstore(documents)

# --- Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# --- Prompt ---
custom_prompt_template = PromptTemplate.from_template("""
You are an expert summarizer and analyst. Given the following chat history and a new user question, provide a precise, confident, and informative answer based strictly on the retrieved transcript context. Do not speculate, hallucinate, or go beyond what’s available. Stay grounded in the excerpts provided.

Instructions for your response:

1. Analyze the nature of the question. Understand what the user is seeking — information, a solution, clarification — and tailor your answer accordingly.
2. Gather all relevant context. Prioritize the most directly relevant excerpts first, then use the rest to support your response.
3. Provide a detailed, context-specific answer. Do not invent or speculate — stick to what the transcript supports.
4. Only include examples if they are available in the transcript.
5. Always answer in first person, as if the question was asked to you directly.
6. If the question is unrelated to the transcript, politely inform the user that you can only answer questions based on the transcript.
7. If the question is ambiguous, ask for clarification.
8. Write naturally — like a human expert, not a bot. Avoid robotic or corporate tones.
9. Quote AJ Wilcox directly when possible to strengthen credibility.
10. Eliminate generic filler — every sentence should deliver value.
11. If multiple transcripts are referenced, list the relevant episode numbers at the end.
12. Please provide detailed but relevant answer, if you have more context available but DO NOT HALLUCINATE.

Chat History:
{chat_history}

Transcript Context:
{context}

Question:
{question}

Answer:
""")

# --- QA Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template},
    output_key="answer"
)

# --- UI ---
st.title("🎙️ What did AJ say?")

st.markdown("Ask me any LinkedIn Ads-related question covered in the podcast transcripts.")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

query = st.text_input("Your question:", key="user_input")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]

        episodes = list({
            doc.metadata.get("episode", "unknown")
            for doc in result.get("source_documents", [])
        })
        reference_note = f"📌 Reference: {', '.join(sorted(episodes))}" if episodes else "📌 Reference: unknown"
        final_answer = f"{answer}\n\n{reference_note}"

        st.session_state.chat_log.append({"user": query, "bot": final_answer})

# --- Chat History ---
for chat in st.session_state.chat_log[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# --- Save button ---
if st.button("💾 Save Chat Log"):
    with open("chat_log.json", "a") as f:
        for entry in st.session_state.chat_log:
            entry["timestamp"] = datetime.now().isoformat()
            f.write(json.dumps(entry) + "\n")
    st.success("Chat log saved!")
