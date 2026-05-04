import os
import uuid
import base64
import streamlit as st
import chromadb
import anthropic
from pdf2image import convert_from_bytes

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

def extract_text_with_claude_vision(pdf_file):
    client = anthropic.Anthropic()
    pdf_bytes = pdf_file.read()
    pages = convert_from_bytes(pdf_bytes, dpi=150)
    
    all_text = ""
    for i, page in enumerate(pages):
        # Convert page image to base64
        import io
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Send to Claude Vision
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL text from this page exactly as it appears. Include text from cards, tables, columns, diagrams and any visual elements. Output only the extracted text, nothing else."
                        }
                    ]
                }
            ]
        )
        
        page_text = message.content[0].text
        all_text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
    
    return all_text

def smart_chunk_text(text, chunk_size=800, overlap=100):
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

def build_collection(chunks):
    db = chromadb.Client()
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
    collection = db.create_collection(collection_name)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def ask_rag(question, collection):
    results = collection.query(query_texts=[question], n_results=4)
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)
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
    return message.content[0].text, retrieved_chunks

st.title("Document Assistant")
st.caption("Upload any PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Reading PDF with Claude Vision... this may take a minute for large documents"):
            text = extract_text_with_claude_vision(uploaded_file)
            chunks = smart_chunk_text(text)
            st.session_state.collection = build_collection(chunks)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.chunk_count = len(chunks)
        st.success(f"Indexed {st.session_state.chunk_count} chunks. Ready for questions!")

    if "messages" in st.session_state and "collection" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            answer, chunks = ask_rag(prompt, st.session_state.collection)
        st.markdown(answer)
        
        # Show retrieved chunks in expander
        with st.expander("📄 View retrieved chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(f"> {chunk}")
                st.divider()
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
