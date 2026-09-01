import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime


# --------------------------------------------------
# 1. EXTRACT TEXT FROM PDF FILES
# --------------------------------------------------

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# --------------------------------------------------
# 2. SPLIT TEXT INTO SMALLER CHUNKS
# --------------------------------------------------

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


# --------------------------------------------------
# 3. CREATE VECTOR STORE USING FAISS
# --------------------------------------------------

def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vector_store


# --------------------------------------------------
# 4. GET ANSWER FROM GEMINI
# --------------------------------------------------

def get_answer(user_question, vector_store, api_key):

    # Find the most relevant text chunks
    docs = vector_store.similarity_search(
        user_question,
        k=4
    )

    # Combine retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the context provided below.

If the answer is not available in the context, clearly say:

"Answer is not available in the provided PDF."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{user_question}

ANSWER:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    response = model.invoke(prompt)

    return response.content


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="📄"
    )

    st.header("DocuMind AI 📄")
    st.write("Upload PDF documents and ask questions about them.")


    # ----------------------------------------------
    # SESSION STATE
    # ----------------------------------------------

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


    # ----------------------------------------------
    # SIDEBAR
    # ----------------------------------------------

    with st.sidebar:

        st.title("Settings ⚙️")

        api_key = st.text_input(
            "Enter your Google Gemini API Key",
            type="password"
        )

        st.markdown(
            "Get your API key from "
            "[Google AI Studio](https://aistudio.google.com/app/apikey)"
        )

        st.divider()

        st.subheader("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )


        # ------------------------------------------
        # PROCESS PDF BUTTON
        # ------------------------------------------

        if st.button("Submit & Process"):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF file."
                )

            else:

                with st.spinner(
                    "Reading and processing PDFs..."
                ):

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():

                        st.error(
                            "Could not extract text from these PDFs."
                        )

                    else:

                        text_chunks = get_text_chunks(
                            raw_text
                        )

                        st.session_state.vector_store = (
                            get_vector_store(text_chunks)
                        )

                        st.session_state.pdf_names = [
                            pdf.name for pdf in pdf_docs
                        ]

                        st.success(
                            "PDFs processed successfully! 🎉"
                        )


        # ------------------------------------------
        # RESET BUTTON
        # ------------------------------------------

        if st.button("Reset Conversation"):

            st.session_state.vector_store = None

            st.session_state.conversation_history = []

            st.success(
                "Conversation has been reset."
            )


    # --------------------------------------------------
    # DISPLAY CONVERSATION HISTORY
    # --------------------------------------------------

    for item in st.session_state.conversation_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])


    # --------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------

    user_question = st.chat_input(
        "Ask a question about your PDFs..."
    )


    if user_question:

        # Check API key
        if not api_key:

            st.warning(
                "Please enter your Google Gemini API key in the sidebar."
            )

            return


        # Check whether PDFs are processed
        if st.session_state.vector_store is None:

            st.warning(
                "Please upload and process your PDF files first."
            )

            return


        # Display user question
        with st.chat_message("user"):

            st.write(user_question)


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = get_answer(
                        user_question,
                        st.session_state.vector_store,
                        api_key
                    )

                    st.write(answer)


                    # Save conversation
                    st.session_state.conversation_history.append(
                        {
                            "question": user_question,
                            "answer": answer,
                            "timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        }
                    )


                except Exception as error:

                    st.error(
                        f"An error occurred: {error}"
                    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    main()