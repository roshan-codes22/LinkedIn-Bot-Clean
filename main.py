import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# LangChain & supporting imports
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Initialize smarter model (70B)
model = ChatGroq(model="llama-3.1-8b-instant")

# Load and prepare transcript
with open("transcript.txt", "r") as f:
    text = f.read()

docs = [Document(page_content=text)]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# Embed and index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding)
db.save_local("transcript_faiss_index")

# Load FAISS index with safety flag
db = FAISS.load_local(
    "transcript_faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Memory buffer for multi-turn conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom system-like prompt
custom_prompt_template = PromptTemplate.from_template("""
You are an expert summarizer and analyst. Given the following chat history and a new user question, provide a precise, concise, and informative answer based only on the transcript. Do not speculate. Stay grounded in the context.

Instructions for you to answer:

1) Always answer the questions in first person as if it is asked to you.
2) If the question is not related to the transcript, politely inform the user that you can only answer questions based on the provided transcript.
3) If the question is ambiguous, ask for clarification.
4) Try to answer as if you are a human being, not a bot.
5) If you do not find the answer here, please reply "I am sorry, it is not in the data as of now."

Chat History:
{chat_history}

Transcript Context:
{context}

Question:
{question}

Answer:
""")

# Conversational QA chain with all upgrades
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": custom_prompt_template}
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
    print("Bot:", answer)

    # Save chat log
    chat_log.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer
    })

# Save to file on exit
with open("chat_log.json", "a") as f:
    for entry in chat_log:
        f.write(json.dumps(entry) + "\n")
