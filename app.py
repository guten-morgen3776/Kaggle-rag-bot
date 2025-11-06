import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os


API_KEY = st.secrets['API_KEY']
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    google_api_key = API_KEY
)

st.title('kaggle解法分析bot')

uploaded_file = st.file_uploader('kaggleの解法PDFをアップロードしてください')
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.path

    st.success(f'{uploaded_file.name}を読み込みました！')

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    st.subheader('PDFから読み取った内容(先頭500文字):')
    st.write(documents[0].page_content[:500])

    user_input = st.txt_input('AIに質問してください(例：kaggleとは？)')

    if user_input:
        with st.spinner("AIが考え中....."):
            response = llm.invoke(user_input)
            st.write(response.content)
        
        os.remove(tmp_file_path)
