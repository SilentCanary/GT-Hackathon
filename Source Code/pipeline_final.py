import os
import torch
import requests
from sentence_transformers import CrossEncoder
from functools import lru_cache
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")
VECTOR_DB_PATH = "FAISS_chunks/faiss_index"

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedder, allow_dangerous_deserialization=True)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

with open("customer_profiles.json", "r", encoding="utf-8") as f:
    CUSTOMER_PROFILES = json.load(f)

with open("store_info.json", "r", encoding="utf-8") as f:
    STORE_INFO = json.load(f)

def groq_call(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": "Use the provided context strictly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content'].strip()

@lru_cache(maxsize=128)
def reformulate_query(prompt: str) -> str:
    reformulate_prompt = f"""
You are an AI assistant tasked with reformulating queries to generate precise, actionable, and personalized answers.
Use all context given to generate a concise and clear response.

Context:
{prompt}

Answer:"""
    return groq_call(reformulate_prompt).strip()

def final_answer_call(reformulated_query: str) -> str:
    final_prompt = f"""
You are an AI assistant. Generate a complete answer to the user query below using the context provided.
Be accurate, concise, and grounded in the chunks only.

User Query + Context:
{reformulated_query}

Answer:"""
    return groq_call(final_prompt).strip()

def query_pipeline(user_query: str, customer_name: str, store_name: str,
                   category_filter=None, top_k=10, rerank_k=5):
    profile = CUSTOMER_PROFILES.get(customer_name, {})
    store = STORE_INFO.get(store_name, {})

    # 🔑 Metadata filtering ensures only relevant chunks
    docs_and_scores = vectorstore.similarity_search_with_score(
        user_query, k=top_k, filter={"customer_name": customer_name}
    )

    retrieved_chunks = [doc.page_content for doc, _ in docs_and_scores]
    if not retrieved_chunks:
        retrieved_chunks = ["No relevant documents found."]

    # Rerank with cross-encoder
    pairs = [(user_query, chunk) for chunk in retrieved_chunks]
    with torch.no_grad():
        scores = cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)

    reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in reranked[:rerank_k]]

    # Build context
    context_text = f"""
User Query: {user_query}

Customer Profile: {profile}

Store Info: {store}

Relevant Document Chunks:
{chr(10).join([f"- {c.strip()}" for c in top_chunks])}
"""

    # Reformulate + Final Answer
    reformulated = reformulate_query(context_text)
    final_answer = final_answer_call(reformulated)
    return final_answer

import gradio as gr

def rag_interface(user_query, customer_name, store_name):
    return query_pipeline(user_query, customer_name, store_name)

if __name__ == "__main__":
    test_cases = [
        {
            "user_query": "Has my recent order been delivered?",
            "customer_name": "User 2",
            "store_name": "Koramangala"
        },
        {
            "user_query": "What is my loyalty tier?",
            "customer_name": "User 5",
            "store_name": "Koramangala"
        },
        {
            "user_query": "Which drinks do I usually order?",
            "customer_name": "User 7",
            "store_name": "Koramangala"
        },
        {
            "user_query": "What are the current offers at my store?",
            "customer_name": "User 3",
            "store_name": "Koramangala"
        },
        {
            "user_query": "Is Cappuccino in stock?",
            "customer_name": "User 4",
            "store_name": "Koramangala"
        }
    ]

    for i, case in enumerate(test_cases, start=1):
        print(f"\n\n=== TEST CASE {i} ===")
        print(f"Query: {case['user_query']}")
        print(f"Customer: {case['customer_name']}")
        print(f"Store: {case['store_name']}")
        print("======================\n")

        answer = query_pipeline(
            user_query=case["user_query"],
            customer_name=case["customer_name"],
            store_name=case["store_name"]
        )

        print(f"\n=== FINAL ANSWER FOR TEST CASE {i} ===")
        print(answer)
        print("=====================================\n")

    with gr.Blocks() as demo:
        gr.Markdown("## Customer Support RAG Demo")

        with gr.Row():
            user_query = gr.Textbox(label="User Query", placeholder="Ask about your order, profile, or store...")
        with gr.Row():
            customer_name = gr.Textbox(label="Customer Name", placeholder="e.g. User 2")
            store_name = gr.Textbox(label="Store Name", placeholder="e.g. Koramangala")

        output = gr.Textbox(label="Answer")

        gr.Button("Get Answer").click(
            fn=rag_interface,
            inputs=[user_query, customer_name, store_name],
            outputs=output
        )

    demo.launch(server_name="127.0.0.1", server_port=0)

