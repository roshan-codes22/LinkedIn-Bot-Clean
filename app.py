import os
import json
import torch
import re
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq # Keep if you might use Groq later
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, AIMessage, HumanMessage # Import these

st.set_page_config(page_title="LinkedIn Podcast Bot", layout="centered")
torch._classes = {}  # Fix for Streamlit + torch

# --- Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- Helper to sanitize $ from user input ---
def sanitize_markdown(text):
    # This might need adjustment if you have valid LaTeX math expressions
    # Simple fix to avoid Streamlit trying to render single $ as math
    return re.sub(r'\$(?![\d\s]*[\+\-\*/=])', r'\\$', text)

# --- Model ---
# Ensure you have GOOGLE_API_KEY set in your .env file
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Load and split documents ---
@st.cache_data
def load_documents():
    import glob
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    # Ensure the 'transcripts' directory exists and contains .txt files
    if not os.path.exists("transcripts"):
         st.error("Error: 'transcripts' directory not found.")
         return [] # Return empty list if directory doesn't exist

    transcript_files = glob.glob("transcripts/*.txt")
    if not transcript_files:
         st.warning("Warning: No transcript files found in 'transcripts' directory.")
         return [] # Return empty list if no files found

    for filepath in transcript_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                episode = os.path.basename(filepath).replace(".txt", "")
                splits = splitter.split_text(text)
                for chunk in splits:
                    documents.append(Document(page_content=chunk, metadata={"episode": episode}))
        except Exception as e:
             st.error(f"Error reading or processing file {filepath}: {e}")
             continue # Skip this file and continue with others
    return documents

# --- Load vectorstore ---
@st.cache_resource
def get_vectorstore(_documents):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Ensure the FAISS index exists before trying to load
    if not os.path.exists("transcript_faiss_index"):
        st.warning("FAISS index not found. Creating a new one (this may take a moment)...")
        if not _documents:
             st.error("Cannot create index: No documents loaded.")
             return None
        try:
            db = FAISS.from_documents(_documents, embedding)
            db.save_local("transcript_faiss_index")
            st.success("FAISS index created and saved.")
            return db
        except Exception as e:
            st.error(f"Error creating FAISS index: {e}")
            return None

    try:
        return FAISS.load_local("transcript_faiss_index", embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.warning("Attempting to recreate index from documents...")
        if not _documents:
             st.error("Cannot recreate index: No documents loaded.")
             return None
        try:
            db = FAISS.from_documents(_documents, embedding)
            db.save_local("transcript_faiss_index")
            st.success("FAISS index recreated and saved.")
            return db
        except Exception as e_recreate:
            st.error(f"Error recreating FAISS index: {e_recreate}")
            return None


documents = load_documents()
db = get_vectorstore(documents)

# Stop if index loading or creation failed
if db is None:
    st.stop()


# --- Memory ---
# Use st.session_state for memory as well, makes saving/loading simpler
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# --- Restore chat memory ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [] # Stores messages in a list of dicts for saving/loading
    st.session_state.memory.chat_memory.messages = [] # Stores messages as Langchain objects for the chain
else:
    # Attempt to load chat log into memory if it exists
    try:
        # Assuming chat_log contains dicts compatible with messages_from_dict
        st.session_state.memory.chat_memory.messages = messages_from_dict(st.session_state.chat_log)
    except Exception as e:
        st.error(f"Error restoring chat memory: {e}")
        st.session_state.chat_log = []
        st.session_state.memory.chat_memory.messages = [] # Clear potentially corrupted memory


# --- Prompt ---
custom_prompt_template = PromptTemplate.from_template("""
You are an expert summarizer and analyst. Given the following chat history and a new user question, your task is to either ask for clarification or provide a detailed answer — but not both at the same time.

Instructions for your response:

1. Analyze the user's question:
   - If the question is too broad, vague, or could benefit from clarification, respond with a follow-up question to understand the user's intent more clearly.
   - If the question is clear and specific, proceed to answer it based strictly on the transcript context.

2. If clarification is needed:
   - Ask only one clarifying question at a time.
   - Do NOT provide an answer until the user has clarified their intent.

3. If the question is clear:
   - Gather all relevant context from the transcript. Prioritize the most relevant excerpts first.
   - Provide a detailed, context-specific answer grounded in the excerpts. Do not speculate or hallucinate.
   - Only include examples if they are explicitly available in the transcript.

4. Always:
   - Answer in first person, as if speaking directly to the user.
   - Quote AJ Wilcox when applicable to strengthen credibility.
   - Avoid robotic or corporate tone — write naturally like a human expert.
   - Eliminate generic filler. Every sentence must add value.
   - If the question is unrelated to the transcript, politely state that.
   - If multiple episodes are referenced, list the relevant episode numbers at the end.

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
    memory=st.session_state.memory, # Use memory from session state
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template},
    output_key="answer"
)

# --- Title ---
st.title("🎙️ What did AJ say?")
st.markdown("Ask me any LinkedIn Ads-related question covered in the podcast transcripts.")

# --- Display chat messages ---
# Iterate through the chat log stored in session state
for msg in st.session_state.chat_log:
    if msg["type"] == "human":
        with st.chat_message("user"):
            st.markdown(sanitize_markdown(msg["data"]["content"]))
    elif msg["type"] == "ai":
        with st.chat_message("assistant"):
            st.markdown(sanitize_markdown(msg["data"]["content"]))

# --- Chat input at the bottom ---
# st.chat_input places the input bar at the bottom of the page
query = st.chat_input("Ask a question about the transcripts...")

# --- Process new message ---
if query:
    # Append user message to chat log immediately
    st.session_state.chat_log.append({"type": "human", "data": {"content": query}})

    # Display the user message using st.chat_message (optional, as the loop above will render it on rerun)
    # with st.chat_message("user"):
    #     st.markdown(sanitize_markdown(query))

    # Use a placeholder or expander for thinking indicator if needed, or just rely on the spinner below
    # with st.chat_message("assistant"):
    #     with st.spinner("Thinking..."): # Spinner inside message bubble

    # Or just use a global spinner:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]

            episodes = list(set(
                doc.metadata.get("episode", "unknown")
                for doc in result.get("source_documents", []) if doc.metadata
            ))

            if "?" in answer:
                final_answer = answer
            else:
                # Check if there are episodes before creating the reference note
                if episodes:
                    reference_note = f"📌 Reference: {', '.join(sorted(episodes))}"
                else:
                    reference_note = "📌 Reference: Context not found in transcripts."
                final_answer = f"{answer}\n\n{reference_note}"

            # Append AI message to chat log
            st.session_state.chat_log.append({"type": "ai", "data": {"content": final_answer}})

            # Display the AI message using st.chat_message (optional, as the loop above will render it on rerun)
            # with st.chat_message("assistant"):
            #      st.markdown(sanitize_markdown(final_answer))

            # Rerun the app to display the new messages
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optional: Log the error to chat history or display it differently
            st.session_state.chat_log.append({"type": "ai", "data": {"content": f"Error processing your request: {e}"}})
            st.rerun()


# --- Save chat log button ---
# Place this outside the chat flow if you want it static, or within the chat area if you want it to scroll
# If you want it static at the bottom, you might need a sidebar or a fixed container (more complex CSS)
# For simplicity, let's place it below the chat input area, it will scroll with the content.
# If you absolutely need it fixed, consider placing it in a sidebar or using HTML/CSS with unsafe_allow_html
if st.button("💾 Save Chat Log"):
    try:
        # Define the log file path
        log_file_path = "chat_logs/chat_log.jsonl" # Use .jsonl for line-delimited JSON

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        with open(log_file_path, "a") as f:
            # Filter out Langchain message objects if they somehow got in (chat_log should contain dicts)
            # And add timestamp for each entry
            log_entries = []
            for entry in st.session_state.chat_log:
                 if isinstance(entry, dict):
                      entry_copy = entry.copy() # Avoid modifying session state directly
                      entry_copy["timestamp"] = datetime.now().isoformat()
                      log_entries.append(entry_copy)
                 # If not a dict, skip or handle as needed

            # Write each entry as a separate JSON line
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

        st.success(f"Chat log saved to {log_file_path}!")
    except Exception as e:
        st.error(f"Error saving chat log: {e}")
