from src.retriever import retrieve, format_context, get_collection
from src.generator import generate


def run_rag_pipeline(question: str,
                     collection,
                     n_results: int = 5,
                     product_filter: str = None) -> dict:
    

    print(f"Searching complaints for: '{question}'")
    retrieved_chunks = retrieve(
        question=question,
        collection=collection,
        n_results=n_results,
        product_filter=product_filter
    )
    print(f"Retrieved {len(retrieved_chunks)} relevant chunks")

    
    context = format_context(retrieved_chunks)

    
    print("Generating answer...")
    result = generate(question=question, context=context)

   
    return {
        "question": question,
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "context_used": context,
        "model": result["model"],
        "tokens_used": result["prompt_tokens"] + result["completion_tokens"]
    }


def display_result(result: dict):
    print(f"QUESTION: {result['question']}")
    print(f"\nANSWER:\n{result['answer']}")
    print("SOURCES USED:")
    for i, chunk in enumerate(result['retrieved_chunks'], 1):
        meta = chunk['metadata']
        print(f"\n  [{i}] Product: {meta.get('product_category')} | "
              f"Issue: {meta.get('issue')} | "
              f"Company: {meta.get('company')} | "
              f"Similarity: {chunk['similarity_score']}")
        print(f"      Text: {chunk['text'][:150]}...")
    print(f"\nTokens used: {result['tokens_used']}")
    