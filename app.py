import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_text(text_chunks,embeddings=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("📂 Retrieve Data from Multiple PDFs 📄✨")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

st.markdown("""
    <style>
        @keyframes shine {
            0% { opacity: 0.5; }
            50% { opacity: 1; filter: drop-shadow(0px 0px 5px rgba(255, 255, 255, 0.8)); }
            100% { opacity: 0.5; }
        }
        .social-icons a:hover img {
            transform: scale(1.2);
        }
            
        .footer-container {
            position: fixed;
            bottom: 0;
            width: 67%;
            justify-content: center;
            text-align: center;            
            padding: 10px 0;
             
        }    

        .divider {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin: 10px ;
        }

        .line {
            flex: 1;
            height: 2px;
            background: white;
            opacity: 0.8;
        }

        .diamond {
            width: 12px;
            height: 12px;
            background: white;
            transform: rotate(45deg);
            box-shadow: 0px 0px 5px rgba(255, 255, 255, 0.8);
            margin: 0 10px;
            animation: shine 2s infinite;
        }

        .footer {
            text-align: center;
            margin-top: 10px;
        }
            
        .social-icons {
            margin-top: 10px;
            display: flex;
            justify-content: center;
            gap: 15px;
        }

         
        @media (max-width: 600px) {
            .social-icons img {
                width: 20px;
                height: 20px;
            }
        }
    </style>

    <div class="footer-container">
        <div class="divider">
            <div class="line"></div>
            <div class="diamond"></div>
            <div class="line"></div>
        </div>

    <div class="footer social-icons">
        <p>👩‍💻:<p>
        <a href="https://twitter.com/ritesh_bhadana" target="_blank">
            <img src="data:image/jpeg;base64,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" width="25">
        </a>
        &nbsp;&nbsp;
        <a href="https://youtube.com/@riteshbhadana" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="25">
        </a>
        &nbsp;&nbsp;
        <a href="https://linkedin.com/in/riteshbhadana" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="25">
        </a>
    </div>
""", unsafe_allow_html=True)




