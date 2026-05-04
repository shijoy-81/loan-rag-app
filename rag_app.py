import os
import streamlit as st
import chromadb
import anthropic
import fitz

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def smart_chunk_text(text, chunk_size=400, overlap=50):
    sections = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for section in sections:
        if len(current_chunk) + len(section) < chunk_size:
            current_chunk += section + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + section + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_collection(chunks, collection_name="uploaded_doc"):
    db = chromadb.Client()
    collection = db.create_collection(collection_name)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def ask_rag(question, collection):
    results = collection.query(query_texts=[question], n_results=2)
    context = "\n\n".join(results['documents'][0])
    prompt = (
        f"Answer the question using ONLY the context below.\n"
        f"If the answer is not in the context, say I don't have that information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

st.title("Document Assistant")
st.caption("Upload any PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Reading and indexing your document..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = smart_chunk_text(text)
            st.session_state.collection = build_collection(chunks)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.chunk_count = len(chunks)
        st.success(f"Document indexed! Created {st.session_state.chunk_count} chunks. Ready for questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer = ask_rag(prompt, st.session_state.collection)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF to get started")
