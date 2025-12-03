# H-002 | Customer Experience Automation
# Track: Customer Experience & Conversational AI

A real-time AI assistant that combines customer data, store inventory, and document knowledge to provide personalized answers to customers.

# 1. The Problem (Real World Scenario)

Context:
Coffee chains, retail stores, and delivery services often have scattered data: customer profiles, order history, store inventory, and operational notes. Account managers or support agents spend hours switching between systems to answer simple customer queries like “Where’s my last order?” or “Which drinks are available today?”

The Pain Point:
Manual retrieval is slow and error-prone. Personalized recommendations or explanations for customers are nearly impossible in real time due to siloed data and unstructured documents.

# Solution:
HyperPersonal AI Agent is a Hybrid RAG (Retrieval-Augmented Generation) system. It unifies structured data (JSON databases) and unstructured data (PDF documents via FAISS) into a single pipeline. using GROQ inference, it delivers concise, context-aware answers instantly.

# 2. Expected End Result

For the User:
```
Input: Ask a query like “Where is my last order?”
```

Action: The system automatically:

- Retrieves relevant info from customer profile, store info, order history, and PDF documents.

- Reranks results using semantic similarity.

- Merges context and generates a clear answer via GROQ.


Output: Receives a personalized answer instantly, e.g.:
```
“Your last order of 2 Caramel Lattes and 1 Hot Chocolate was delivered from Koramangala store at 11:42 AM. Contactless delivery was used.”
```

# 3. Technical Approach

## System Architecture

- **Structured DBs (JSON)**
  - Customer Profiles (`customer_profiles.json`)
  - Store Info (`store_info.json`)
  - Orders, Inventory, Coupons → prototyped in separate JSON files

- **Document Retrieval (FAISS + Embeddings)**
  - Use LangChain to load PDFs, chunk them, and store embeddings in FAISS
  - Embedding model: `all-MiniLM-L6-v2`
  - Cross-Encoder reranking: `ms-marco-MiniLM-L-12-v2`

- **Context Merging**
  - Combine: user query + structured DB info + top-k retrieved document chunks
  - add filtering by customer/store

- **GROQ Inference**
  - Reformulates and generates final answers
  - Ensures the tone matches official documentation and avoids hallucinations


# End-to-End Flow:

- **User Query**  
- ↓ Identify Customer / Store  
- ↓ FAISS Retrieval + Top-K Rerank  
- ↓ Merge Context (DB + Documents + Intent)  
- ↓ Reforumulate the final query
- ↓ LLM generates the response
- ↓ Return Answer

# 4. Tech Stack

Language: Python 3.11

Vector DB: FAISS

Embedding & Rerank: SentenceTransformers + CrossEncoder

LLM: GROQ (meta-llama/llama-4-scout-17b-16e-instruct)

Document Processing: LangChain + PyPDFLoader + RecursiveCharacterTextSplitter

Data Storage: JSON for prototyping


# 5. Challenges & Learnings

Challenge 1: Document + Structured Data Merging

Issue: Different PDF types and JSONs made merging complex.

Solution: Added metadata (customer_name, store_name, category) during ingestion and filtered retrieval accordingly.

Challenge 2: LLM Hallucinations

Issue: Early versions produced answers not grounded in customer/order data.

Solution: Merged context and used strict system prompts for GROQ to always rely on provided data.

Challenge 3: Multi-Customer Handling

Issue: Initially, hardcoded dictionaries only supported one customer/store.

Solution: Switched to JSON databases for easy scaling.

# 6. Visual Proof
<img width="602" height="186" alt="image" src="https://github.com/user-attachments/assets/b910e5a4-72ee-4ada-82db-3d54646c5c4e" />


# 7. How to Run
# 1. Clone Repository
```bash
git clone https://github.com/username/hyperpersonal-agent.git
```
# 2. Add API Key
```bash
export GROQ_KEY="your_key_here"
```

# 4. Test
for creating the chunks and storing to DB (faiss)
```bash
python chunking.py
```

for running pipeline 
```bash
python pipeline.py
```

# Example:
```
 User query: "Where is my last order?"
```
