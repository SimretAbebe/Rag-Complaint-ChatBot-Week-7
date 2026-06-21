import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant" 
MAX_TOKENS = 1024


PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your job is to analyze customer complaints and provide clear, 
evidence-based answers to questions from internal teams.

STRICT RULES:
1. Answer ONLY based on the complaint excerpts provided below
2. NEVER make up information not found in the context
3. If the context does not contain enough information, say so clearly
4. Always mention which products or companies are involved
5. Be concise and professional — this is for internal business use

RETRIEVED COMPLAINT EXCERPTS:
{context}

QUESTION: {question}

ANSWER (based only on the complaints above):"""


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


def generate(question: str, context: str) -> dict:
    prompt = build_prompt(question, context)

    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful financial analyst. Answer based only on provided complaint data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1  
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "model": GROQ_MODEL,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens
    }