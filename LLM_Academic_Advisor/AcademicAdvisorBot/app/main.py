import streamlit as st
from scripts.build_chain import build_chain

st.set_page_config(page_title="🎓 Academic Advisor Chatbot")
st.title("🎓 Academic Advisor Chatbot")

if "chain" not in st.session_state:
    st.session_state.chain = build_chain()

query = st.text_input("Ask me anything academic:")

if query:
    with st.spinner("Thinking..."):
        response = st.session_state.chain.run(query)
        st.write(response)
