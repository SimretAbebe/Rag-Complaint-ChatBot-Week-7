# CrediTrust RAG Complaint Chatbot

An AI-powered internal tool that transforms raw customer complaint data into instant, evidence-backed insights for product, support, and compliance teams.

## Business Problem

CrediTrust Financial receives thousands of complaints monthly across 4 products.
Teams spend days manually reading complaints to find patterns.
This tool lets anyone ask plain-English questions and get answers in seconds.

## Solution

A Retrieval-Augmented Generation (RAG) chatbot that:

- Searches 465,679 real CFPB complaints semantically
- Retrieves the most relevant complaint narratives
- Generates concise, evidence-backed answers via an LLM

## Products Covered

| Product         | Complaints |
| --------------- | ---------- |
| Credit Card     | 189,334    |
| Savings Account | 140,319    |
| Money Transfer  | 98,685     |
| Personal Loan   | 37,341     |

## Project Structure

```
Week-7/

├── data/
│   ├── raw/              - Raw CFPB dataset
│   └── processed/        - Cleaned/filtered dataset

├── notebooks/            - Jupyter notebooks per task

├── src/                  - Source modules

├── tests/                - Unit tests

├── vector_store/         - FAISS/ChromaDB index

└── app.py                - Streamlit/Gradio UI
```
