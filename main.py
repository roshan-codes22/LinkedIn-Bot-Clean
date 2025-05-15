import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# LangChain & supporting imports
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Initialize smarter model (Groq LLaMA or Gemini)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Load transcript documents
transcript_folder = "transcripts"

def load_documents_from_folder(transcript_folder):
    documents = []
    for filename in os.listdir(transcript_folder):
        if filename.endswith(".txt"):
            episode = filename.replace(".txt", "")
            with open(os.path.join(transcript_folder, filename), "r", encoding='utf-8') as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"episode": episode})
                documents.append(doc)
    return documents

# Split documents while preserving metadata
def split_docs_with_metadata(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for chunk in splits:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs

# Prepare documents
docs = load_documents_from_folder(transcript_folder)
documents = split_docs_with_metadata(docs)

# Embed and index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding)
db.save_local("transcript_faiss_index")

# Load FAISS index
db = FAISS.load_local(
    "transcript_faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Memory buffer for multi-turn conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Custom system prompt
custom_prompt_template = PromptTemplate.from_template("""
You are an expert summarizer and analyst. Given the following chat history and a new user question, your task is to either ask for clarification or provide a detailed answer — but not both at the same time.

Instructions for your response:

1. First, analyze the user's question:
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

# Conversational QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template},
    output_key="answer"
)

# Start chat loop
print("🤖 Smart Bot is ready. Type your question (or 'exit' to quit):\n")
chat_log = []

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    result = qa_chain.invoke({"question": query})
    answer = result["answer"]

    # Extract referenced episodes
    episodes = list({
        doc.metadata.get("episode", "unknown")
        for doc in result.get("source_documents", [])
    })

    # Add reference only if it's not a follow-up question
    if "?" not in answer:
        reference_note = f"📌 Reference: {', '.join(sorted(episodes))}" if episodes else "📌 Reference: unknown"
        final_answer = f"{answer}\n\n{reference_note}"
    else:
        final_answer = answer

    print("Bot:", final_answer)

    chat_log.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": final_answer
    })

# Save to file on exit
with open("chat_log.json", "a") as f:
    for entry in chat_log:
        f.write(json.dumps(entry) + "\n")
