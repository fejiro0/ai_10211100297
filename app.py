# Name: FAMOUS AKPOVOGBETA
# Index Number: 10211100297

import streamlit as st

rag_app = st.Page("streamlit_files/rag_app.py", title="Exam RAG Chatbot")
pg = st.navigation([rag_app])
pg.run()
