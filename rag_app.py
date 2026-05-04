import os
import streamlit as st
import chromadb
import anthropic

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

pdf_text = (
    "Loan Application. ABC Company 401(k) Plan Name. Plan Number: 600001. "
    "Employee Full Name: Sally Smith. SSN: 555-55-5555. "
    "Permanent Address: 1033 E. Griffith Way, Fresno, CA 93704. "
    "Email: sallysmith@gmail.com. Phone: 651-333-4290. "
    "Purpose: Debt. Amount: Maximum amount allowed. Duration: 60 months / 5 years.\n\n"
    "Loan Information Notes: Some plans only allow loans for hardship reasons. "
    "See the Loan Policy for any loan limitations. "
    "If the Plan allows one outstanding loan at a time, payoff of the first loan "
    "must be made prior to requesting another loan.\n\n"
    "Irrevocable Pledge and Assignment: "
    "The employee irrevocably pledges 50% of vested account balances to the Trustee. "
    "Failure to repay authorizes the Trustee to foreclose on this security. "
    "The employee enters into a payroll deduction arrangement to repay the loan in full.\n\n"
    "Fees and Conditions: "
    "1. A setup fee applies. "
    "2. An annual loan maintenance fee may apply. "
    "3. Certain plans subject the loan balance to asset based fees. "
    "4. An overnight fee applies if overnight delivery is requested. "
    "5. Alerus will withhold Florida document excise tax for Florida residents. "
    "Employee Signature Date: 5/19/25.\n\n"
    "Authorized Signature Section: For Employer or Authorized Party use only. "
    "Payroll frequency: Weekly. First payment date: 6-20-2025. "
    "Vesting: based on participant vesting percentage as most recently reported. "
    "Loan interest rate: Prime Rate as posted in the Wall Street Journal plus 1%. "
    "Authorized Signature Date: 5-19-25. "
    "Submit completed form to Alerus Retirement Solutions via Plan Gateway "
    "at alerusretirementsolutions.com or by mail to "
    "Two Pine Tree Drive, Suite 400, Arden Hills, MN 55112."
)

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

@st.cache_resource
def setup_rag():
    chunks = smart_chunk_text(pdf_text)
    db = chromadb.Client()
    collection = db.create_collection("loan_app")
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

st.title("Loan Document Assistant")
st.caption("Ask me anything about the ABC Company 401(k) loan application")

if "messages" not in st.session_state:
    st.session_state.messages = []

collection = setup_rag()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the loan document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            answer = ask_rag(prompt, collection)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
