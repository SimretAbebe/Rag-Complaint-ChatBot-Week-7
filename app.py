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
        return (
            "Please type a question first.",
            "",
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
        )
    
    
    filter_value = None if product_filter == "All Products" else product_filter
    
    try:
       
        chunks = retrieve(
            question=question,
            collection=collection,
            n_results=num_sources,
            product_filter=filter_value
        )
        
        
        if not chunks:
            # no results found — show empty message area
            return (
                "",  # answer output empty
                "",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(value="No relevant complaints found for that query and filter.", visible=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
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
        
        # re-enable Ask button and inputs after processing and hide spinner
        return (
            answer,
            sources_text,
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    
    except Exception as e:
        return (
            f"An error occurred: {str(e)}",
            "",
            gr.update(interactive=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    


def clear_all():
    # reset inputs, hide spinner, and ensure Ask button is enabled
    return (
        "",
        "All Products",
        5,
        "",
        "",
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def set_loading():
    # show spinner, clear previous outputs, and disable Ask button
    return (
        "",
        "",
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(value="", visible=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


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

            # Quick product presets for non-technical users
            with gr.Row():
                pc_credit = gr.Button("Credit Card", elem_id="preset-credit")
                pc_savings = gr.Button("Savings Account", elem_id="preset-savings")
                pc_transfer = gr.Button("Money Transfer", elem_id="preset-transfer")

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

            # spinner / loading indicator (hidden by default)
            spinner_html = gr.HTML(
                "<div style='display:flex;align-items:center;gap:8px'><div style=\"border:4px solid #f3f3f3;border-top:4px solid #444;border-radius:50%;width:18px;height:18px;animation:spin 1s linear infinite;\"></div><div style='color:#999'>Retrieving...</div></div>",
                visible=False,
            )

            gr.Markdown("### Sources Used\n\n")
            sources_output = gr.Markdown(
                label="Sources",
                value="*Sources will appear here.*"
            )
            
            # improved empty result message area (hidden when not used)
            empty_message = gr.Markdown(value="", visible=False)

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
    # show immediate loading text (disables button), then run the heavy retrieval/generation
    ask_btn.click(
        fn=set_loading,
        inputs=[],
        outputs=[answer_output, sources_output, ask_btn, spinner_html, empty_message, product_filter, num_sources, question_input]
    )
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output, ask_btn, spinner_html, empty_message, product_filter, num_sources, question_input]
    )

    # preset buttons set the product filter value
    pc_credit.click(lambda: "Credit Card", inputs=[], outputs=[product_filter])
    pc_savings.click(lambda: "Savings Account", inputs=[], outputs=[product_filter])
    pc_transfer.click(lambda: "Money Transfer", inputs=[], outputs=[product_filter])

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, product_filter, num_sources,
                 answer_output, sources_output, ask_btn, spinner_html, empty_message, product_filter, num_sources, question_input]
    )

   
    # submit should mirror the button behaviour (loading then ask)
    question_input.submit(
        fn=set_loading,
        inputs=[],
        outputs=[answer_output, sources_output, ask_btn, spinner_html, empty_message, product_filter, num_sources, question_input]
    )
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output, ask_btn, spinner_html, empty_message, product_filter, num_sources, question_input]
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