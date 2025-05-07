import os
import json
import torch
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


st.set_page_config(page_title="LinkedIn Podcast Bot", layout="centered")
torch._classes = {}  # hotfix for Streamlit + torch class issue on Mac

# --- SETUP ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- Load model ---
model = ChatGroq(model="llama-3.1-8b-instant")

# --- Load transcript only once ---
@st.cache_data
def load_documents():
    with open("transcript.txt", "r") as f:
        text = f.read()
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- Create or load FAISS index ---
@st.cache_resource
def get_vectorstore(_documents):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding)
    db.save_local("transcript_faiss_index")
    return FAISS.load_local("transcript_faiss_index", embedding, allow_dangerous_deserialization=True)

documents = load_documents()
db = get_vectorstore(documents)

# --- Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Prompt ---
custom_prompt_template = PromptTemplate.from_template("""
You are an expert summarizer and analyst. Given the following chat history and a new user question, provide a precise, concise, and informative answer based only on the transcript. Do not speculate. Stay grounded in the context.

Instructions for you to answer:

1) Always answer the questions in first person as if it is asked to you.
2) If the question is not related to the transcript, politely inform the user that you can only answer questions based on the provided transcript.
3) If the question is ambiguous, ask for clarification.
4) Keep the answer crisp and confident.
5) Try to answer as if you are a human being, not a bot.

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
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template}
)

# --- Streamlit UI ---
st.title("🎙️ LinkedIn QA Bot")

st.markdown("Ask me and I shall answer!")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

query = st.text_input("Your question:", key="user_input")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        st.session_state.chat_log.append({"user": query, "bot": answer})

# --- Show chat history ---
for chat in st.session_state.chat_log[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# --- Save on exit (optional) ---
if st.button("💾 Save Chat Log"):
    with open("chat_log.json", "a") as f:
        for entry in st.session_state.chat_log:
            entry["timestamp"] = datetime.now().isoformat()
            f.write(json.dumps(entry) + "\n")
    st.success("Chat log saved!")