import streamlit as st
import os
from operator import itemgetter
from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import PyPDF2


def process_upload_file(uploaded_file):
    # Step 1 - Read file
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    # Step 2 - Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Step 3 - Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 4 - Convert and store in ChromaDB
    chunks = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Step 5 - Build chain
    total_chunks = len(chunks)

    if total_chunks <= 5:
        k = 2
    elif total_chunks <= 15:
        k = 4
    elif total_chunks <= 30:
        k = 6
    else:
        k = 8
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    print(f"Document has {total_chunks} chunks -> using k = {k}")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=500,
    )

    # Generic prompt — works for ANY document, not specific to one
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent document analyzer.
    You can read and answer questions about ANY type of document.
                                              
    YOUR RULES:
    1. Answer ONLY from the context provided
    2. If answer not in context -> say "I don't have that information in the provided document."
    3. If calculation needed -> show your working step by step
    4. Match answer style to document type
    5. Today's date is {today} - use this for any date calculations
                                              
    ANSWER FORMAT:
    [Direct answer to the question]
    [Supporting detail from document if relevant]

    Previous conversation:
    {chat_history}
                     
    Context from document: 
    {context}

    Question: {question}
    Answer: """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "today": itemgetter("today"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Page Setup ─────────────────────────────────────────────────────
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📄 AI Document Assistant")

# ── Session State ──────────────────────────────────────────────────
# messages     — stores conversation history
# chain        — stores the built RAG chain
# current_file — tracks which file is currently loaded
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None


# --- Chat history
def get_chat_history():
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role} : {msg['content']}\n"
    return history


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Document")

    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    # File handling logic:
    if uploaded_file is not None:
        # NEW file uploaded — process it and clear old chat
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing document..."):
                st.session_state.chain = process_upload_file(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []  # ← clear old chat automatically
            st.success(f"✅ {uploaded_file.name} ready!")
        else:
            # Same file — just show its name, don't reprocess
            st.info(f"📄 {uploaded_file.name}")
    else:
        # File removed — clear everything automatically
        if st.session_state.current_file is not None:
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.current_file = None

    st.divider()

    # Clear button — clears chat but keeps file loaded
    if st.session_state.chain is not None:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

# ── Main Chat Area ─────────────────────────────────────────────────
if st.session_state.chain is None:
    st.info("👈 Upload a document from the sidebar to get started!")
else:
    # Show conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input box
    question = st.chat_input("Ask me anything about your document...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # now passing both question and chat history
                answer = st.session_state.chain.invoke(
                    {
                        "question": question,
                        "chat_history": get_chat_history(),
                        "today": date.today().strftime("%B %d %Y"),
                    }
                )
            st.write(answer)

        # Save answer to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
