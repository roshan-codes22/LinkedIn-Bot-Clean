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

    reference_note = f"📌 Reference: {', '.join(sorted(episodes))}" if episodes else "📌 Reference: unknown"
    final_answer = f"{answer}\n\n{reference_note}"
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
