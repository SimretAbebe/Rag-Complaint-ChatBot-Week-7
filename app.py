import os
import sys
import gradio as gr
from dotenv import load_dotenv

CSS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
    "styles.css"
)

print(f"CSS path: {CSS_PATH}")
print(f"CSS file exists: {os.path.exists(CSS_PATH)}")

with open(CSS_PATH, "r") as f:
    CUSTOM_CSS = f.read()

print(f"CSS loaded: {len(CUSTOM_CSS)} characters")

from src.retriever import retrieve, format_context, get_collection
from src.generator import generate

load_dotenv()

PRODUCT_CHOICES = [
    "All Products",
    "Credit Card",
    "Savings Account",
    "Money Transfer",
    "Personal Loan"
]

print("Loading RAG components...")

collection = get_collection()
print(f"Vector store loaded: {collection.count():,} chunks ready")

print("RAG system ready!")


def ask_question(question: str,
                 product_filter: str,
                 num_sources: int) -> tuple:
   
    
  
    if not question or question.strip() == "":
        return "Please type a question first.", ""
    
    
    filter_value = None if product_filter == "All Products" else product_filter
    
    try:
       
        chunks = retrieve(
            question=question,
            collection=collection,
            n_results=num_sources,
            product_filter=filter_value
        )
        
        
        context = format_context(chunks)
        
        
        result = generate(question=question, context=context)
        answer = result['answer']
        
       
        sources_text = ""
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            sources_text += (
                f"**Source {i}**\n"
                f"- **Product:** {meta.get('product_category', 'Unknown')}\n"
                f"- **Issue:** {meta.get('issue', 'Unknown')}\n"
                f"- **Company:** {meta.get('company', 'Unknown')}\n"
                f"- **Similarity Score:** {chunk['similarity_score']}\n"
                f"- **Excerpt:** *{chunk['text'][:200]}...*\n\n"
                f"---\n\n"
            )
        
        return answer, sources_text
    
    except Exception as e:
        return f"An error occurred: {str(e)}", ""
    


def clear_all():
    return "", "All Products", 5, "", ""


# Building the interface
with gr.Blocks(
    title="CrediTrust Complaint Chatbot",
) as app:

    # Header
    gr.HTML("""
    <div style="
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;">
        <h1 style="color:#ffffff; font-size:1.8rem; font-weight:700; margin:0 0 8px 0;">
            CrediTrust Financial
        </h1>
        <p style="color:#888888; font-size:0.95rem; margin:0;">
            Complaint Analysis Chatbot — Ask plain-English questions 
            about 464,000+ real customer complaints
        </p>
    </div>
    """)

    with gr.Row():

       
        with gr.Column(scale=1):
            gr.Markdown("### Settings")

            product_filter = gr.Dropdown(
                choices=PRODUCT_CHOICES,
                value="All Products",
                label="Filter by Product",
                info="Narrow results to a specific product"
            )

            num_sources = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of Sources",
                info="How many complaint chunks to retrieve"
            )

        # Right column side main chat area
        with gr.Column(scale=3):
            gr.Markdown("Ask a Question")

            question_input = gr.Textbox(
                placeholder="e.g. Why are customers unhappy with credit cards?",
                label="Your Question",
                lines=3
            )

            with gr.Row():
                ask_btn = gr.Button(
                    "Ask Question",
                    variant="primary"
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary"
                )

            gr.Markdown("Answer")
            answer_output = gr.Markdown(
                label="Answer",
                value="*Your answer will appear here.*"
            )

            gr.Markdown("### Sources Used\n\n")
            sources_output = gr.Markdown(
                label="Sources",
                value="*Sources will appear here.*"
            )

    # Example questions
    gr.Markdown("### Example Questions")
    gr.Examples(
        examples=[
            ["Why are customers unhappy with credit cards?", "Credit Card", 5],
            ["What are the most common issues with money transfers?", "Money Transfer", 5],
            ["What fraud issues do customers report?", "All Products", 5],
            ["Why do customers complain about savings accounts?", "Savings Account", 5],
            ["What problems do customers face with personal loans?", "Personal Loan", 5],
        ],
        inputs=[question_input, product_filter, num_sources],
        label="Click any example to try it"
    )

    # Connect buttons to functions
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, product_filter, num_sources,
                 answer_output, sources_output]
    )

   
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output]
    )
    
    # Footer
    gr.HTML("""
    <div style="
        text-align:center;
        padding:20px;
        color:#555555;
        font-size:0.8rem;
        border-top:1px solid #222222;
        margin-top:20px;">
        CrediTrust Financial — RAG Complaint Analysis System &nbsp;|&nbsp;
        LangChain · ChromaDB · Groq LLM · Gradio &nbsp;|&nbsp;
        Data: CFPB Public Complaints Dataset
    </div>
    """)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )