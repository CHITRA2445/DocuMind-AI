import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


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
                text += page_text + "\n"

    return text


# --------------------------------------------------
# 2. SPLIT TEXT INTO SMALLER CHUNKS
# --------------------------------------------------

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)


# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 4. CREATE VECTOR STORE USING FAISS
# --------------------------------------------------

def get_vector_store(text_chunks):

    embeddings = load_embeddings()

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vector_store


# --------------------------------------------------
# 5. CLEAN MODEL RESPONSE
# --------------------------------------------------

def extract_response_text(response):
    """
    Extract only the actual text from the model response.
    Ignores metadata, signatures, SVG data, etc.
    """

    content = response.content

    # Normal text response
    if isinstance(content, str):
        return content.strip()

    # Content returned as a list
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                # Only extract actual text blocks
                if item.get("type") == "text":
                    text = item.get("text")

                    if text:
                        text_parts.append(text)

        return "\n".join(text_parts).strip()

    # Fallback
    return str(content).strip()


# --------------------------------------------------
# 6. GET ANSWER FROM GEMINI
# --------------------------------------------------

def get_answer(user_question, vector_store, api_key):

    # Find relevant chunks
    docs = vector_store.similarity_search(
        user_question,
        k=4
    )

    # Combine retrieved document content
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are DocuMind AI, a helpful document question-answering assistant.

Answer the user's question ONLY using the context provided below.

If the answer is not available in the provided context, clearly say:

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

    return extract_response_text(response)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    # Read API key securely from Streamlit Secrets
    api_key = st.secrets["GOOGLE_API_KEY"]

    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="📄",
        layout="wide"
    )

    st.title("DocuMind AI 📄")

    st.write(
        "Upload PDF documents and ask questions based on their content."
    )


    # --------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "pdf_names" not in st.session_state:
        st.session_state.pdf_names = []


    # --------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------

    with st.sidebar:

        st.title("DocuMind AI")

        st.caption(
            "Upload your PDF documents and ask questions about them."
        )

        st.divider()

        st.subheader("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )


        # --------------------------------------------------
        # PROCESS PDF BUTTON
        # --------------------------------------------------

        if st.button(
            "Submit & Process",
            use_container_width=True
        ):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF file."
                )

            else:

                with st.spinner(
                    "Reading and processing PDFs..."
                ):

                    try:

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

                            # Clear previous conversation
                            st.session_state.conversation_history = []

                            st.success(
                                "PDFs processed successfully! 🎉"
                            )

                    except Exception as error:

                        st.error(
                            f"Error while processing PDFs: {error}"
                        )


        # --------------------------------------------------
        # DISPLAY PROCESSED FILES
        # --------------------------------------------------

        if st.session_state.pdf_names:

            st.divider()

            st.subheader("Processed Files")

            for file_name in st.session_state.pdf_names:

                st.write(f"📄 {file_name}")


        # --------------------------------------------------
        # RESET BUTTON
        # --------------------------------------------------

        st.divider()

        if st.button(
            "Reset Conversation",
            use_container_width=True
        ):

            st.session_state.vector_store = None

            st.session_state.conversation_history = []

            st.session_state.pdf_names = []

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

                    # Display only clean answer
                    st.write(answer)


                    # Save conversation
                    st.session_state.conversation_history.append(
                        {
                            "question": user_question,
                            "answer": answer
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