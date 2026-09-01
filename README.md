# 📄 DocuMind AI

DocuMind AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The application extracts text from uploaded PDFs, splits the content into smaller chunks, converts them into vector embeddings, and stores them in a FAISS vector database. When a user asks a question, the most relevant document chunks are retrieved and provided to Google Gemini to generate a context-aware answer.

---

## 🚀 Features

- 📄 Upload one or multiple PDF documents
- 🔍 Extract text from PDF files
- ✂️ Split document text into smaller chunks
- 🧠 Generate embeddings using Hugging Face models
- 📦 Store embeddings using FAISS vector database
- 🔎 Retrieve relevant document chunks using similarity search
- 🤖 Generate answers using Google Gemini
- 💬 Chat-based user interface using Streamlit
- 📝 Maintain conversation history during the session
- 🔐 Secure API key management using Streamlit Secrets
- 🔄 Reset conversation and document data

---

## 🏗️ How It Works

```text
Upload PDF Documents
        ↓
Extract Text Using PyPDF2
        ↓
Split Text into Chunks
        ↓
Generate Embeddings
        ↓
Store Embeddings in FAISS
        ↓
User Asks a Question
        ↓
Retrieve Relevant Chunks
        ↓
Send Context + Question to Gemini
        ↓
Generate Context-Aware Answer
```

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Hugging Face**
- **Sentence Transformers**
- **FAISS**
- **Google Gemini**
- **PyPDF2**

---

## 📂 Project Structure

```text
DocuMind-AI/
│
├── .streamlit/
│   └── secrets.toml
│
├── faiss_index/
│
├── app.py
├── main.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── langchain_rag_chatbot.ipynb
├── rag_pipeline.ipynb
└── rag_pipelines.ipynb
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CHITRA2445/DocuMind-AI.git
```

Move into the project directory:

```bash
cd DocuMind-AI
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```



---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

You can then:

1. Upload one or more PDF files.
2. Click **Submit & Process**.
3. Wait for the PDFs to be processed.
4. Ask questions about the uploaded documents.
5. Receive AI-generated answers based on the relevant PDF content.

---

## 🧠 RAG Pipeline

DocuMind AI follows a Retrieval-Augmented Generation workflow.

### Step 1: PDF Text Extraction

Text is extracted from uploaded PDF documents using `PyPDF2`.

### Step 2: Text Chunking

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

This improves document retrieval and ensures relevant information can be efficiently processed.

### Step 3: Embedding Generation

Text chunks are converted into vector embeddings using the Hugging Face model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Step 4: Vector Storage

The generated embeddings are stored in a FAISS vector store.

### Step 5: Similarity Search

When the user asks a question, the application performs similarity search to retrieve the most relevant document chunks.

### Step 6: Answer Generation

The retrieved document context and the user's question are sent to Google Gemini.

The model is instructed to answer only using the provided context and avoid generating unsupported information.

---

## 💻 Requirements

The main dependencies include:

```text
streamlit
langchain
faiss-cpu
langchain-core
langchain-google-genai
PyPDF2
google-ai-generativelanguage
langchain-community
pandas
langchain-huggingface
wikipedia
sentence-transformers
transformers
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📸 Application Workflow

```text
Open DocuMind AI
       ↓
Upload PDF Documents
       ↓
Submit & Process PDFs
       ↓
Ask a Question
       ↓
Relevant Context Retrieved
       ↓
Get AI-Generated Answer
```





## 👩‍💻 Author

**Chitra Singh**

GitHub: https://github.com/CHITRA2445

---


