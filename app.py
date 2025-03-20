import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# Make sure to set OPENAI_API_KEY in your .env file

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()  # Using OpenAI embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Using OpenAI's gpt-4o-mini model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()  # Using OpenAI embeddings
    
    # Added allow_dangerous_deserialization=True to fix the ValueError
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")
    
    # Add a warning about the FAISS deserialization
    if os.path.exists("faiss_index"):
        st.info("Using saved vector index. This requires allowing potentially unsafe deserialization.")
    
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        if os.path.exists("faiss_index"):
            user_input(user_question)
        else:
            st.error("Please upload and process PDFs first before asking questions.")
    
    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."): # user friendly message.
                    raw_text = get_pdf_text(pdf_docs) # get the pdf text
                    text_chunks = get_text_chunks(raw_text) # get the text chunks
                    get_vector_store(text_chunks) # create vector store
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")
        
        st.write("---")
        st.write("AI App created by @ Arya Ajay Gupta")
    
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/thegupta1694/ChatPDF" target="_blank">Arya Ajay Gupta</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
