# CrediTrust RAG Complaint Chatbot

An AI-powered internal tool that transforms raw customer complaint data
into instant, evidence-backed insights for product, support, and compliance teams.

---

## Business Problem

CrediTrust Financial receives thousands of customer complaints every month
across four financial products. Teams spend days manually reading complaints
to find patterns. By the time they have answers, problems have already escalated.

This tool lets anyone ask plain-English questions and get clear,
evidence-backed answers in seconds — no technical skills required.

---

## Solution

A Retrieval-Augmented Generation (RAG) chatbot that:

- Accepts plain-English questions from internal teams
- Searches real CFPB complaint narratives using semantic search
- Retrieves the most relevant complaint chunks from a vector database
- Generates concise, evidence-backed answers using a large language model
- Displays source complaints so users can verify every answer

---

## How It Works

User asks a question

↓

Question converted to embedding (all-MiniLM-L6-v2)

↓

ChromaDB searches for most relevant complaint chunks

↓

Retrieved chunks injected into prompt template

↓

Groq LLM (Llama 3) generates a grounded answer

↓

Answer + sources displayed in Gradio UI

```
---

## Products Covered

- Credit Card
- Savings Account
- Money Transfer
- Personal Loan

---

## Project Structure
```

rag-complaint-chatbot/

├── data/

│ ├── raw/ # Raw CFPB dataset

│ └── processed/ # Cleaned and filtered dataset

├── notebooks/

│ ├── task1_eda_preprocessing.ipynb

│ └── task2_chunking_embedding.ipynb

├── src/

│ ├── retriever.py # Semantic search function

│ ├── generator.py # Groq LLM answer function

│ ├── rag_pipeline.py # End-to-end pipeline

│ └── styles.css # Custom UI styling

├── tests/

│ ├── test_preprocessing.py # Task 1 unit tests

│ ├── test_chunking.py # Task 2 unit tests

│ └── test_rag_pipeline.py # Task 3 unit tests

├── vector_store/ # Persisted ChromaDB index

├── app.py # Gradio UI

├── requirements.txt # Project dependencies

└── .github/

└── workflows/

└── unittests.yml # CI/CD pipeline

````
---

## Tech Stack

| Component | Tool |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 (Sentence Transformers) |
| Vector Database | ChromaDB |
| LLM | Llama 3 via Groq API |
| RAG Framework | LangChain |
| UI | Gradio |
| Testing | pytest + GitHub Actions |
| Data Source | CFPB Public Complaints Dataset |

---

## Setup

```bash
git clone https://github.com/SimretAbebe/Rag-Complaint-ChatBot-Week-7.git
cd Rag-Complaint-ChatBot-Week-7
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
````

Create a `.env` file at the project root:

```

GROQ_API_KEY=your_key_here

```

Run the app:

```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## Project Progress

- [x] Task 1 — EDA and Data Preprocessing
- [x] Task 2 — Text Chunking and Embedding Pipeline
- [x] Task 3 — RAG Core Logic and Evaluation
- [x] Task 4 — Gradio UI

---
