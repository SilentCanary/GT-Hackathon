# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:19:00 2025

@author: advit
"""

import fitz  # pip install PyMuPDF
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "customer pdfs"
VECTOR_DB_PATH = "FAISS_chunks/faiss_index"
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CATEGORY_MAP = {
    "customer_profiles_realistic.pdf": "CustomerProfile",
    "order_status_realistic.pdf": "OrderTracking",
    "store_info_realistic.pdf": "StoreInfo",
    "inventory_realistic.pdf": "Inventory"
}

def safe_extract(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def entity_to_text(entity: dict) -> str:
    """Convert dict entity into natural language text for embeddings."""
    return "\n".join([f"{k.replace('_',' ').title()}: {v}" for k, v in entity.items() if v])

# === Extraction functions ===
def extract_customer_profiles(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    customers = []
    blocks = re.split(r'Customer ID:\s*CUST-\d+', full_text)[1:]
    for block in blocks:
        name = safe_extract(r'Name:\s*(User \d+)', block)
        if not name: continue
        customer = {
            "customer_name": name,
            "loyalty_tier": safe_extract(r'Loyalty Tier:\s*([A-Za-z]+)', block),
            "phone": safe_extract(r'Phone:\s*(\+[^\n]+)', block),
            "favorite_drinks": safe_extract(r'Favourite Drinks:\s*([^\n]+)', block),
            "addresses": safe_extract(r'Addresses:\s*(.+)', block),
            "special_notes": safe_extract(r'Special Notes:\s*(.+)', block)
        }
        customers.append(customer)
    return customers

def extract_orders(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    orders = []
    pattern = r'Order #(\d+)\s+Customer:\s*(User \d+)\s+Order Date:\s*([^\n]+).*?Status:\s*([^\n]+)'
    matches = re.finditer(pattern, full_text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        order_num, customer, date, status = match.groups()
        orders.append({
            "order_number": f"#{order_num}",
            "customer_name": customer.strip(),
            "order_date": date.strip(),
            "status": status.strip()
        })
    return orders

def extract_stores(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    stores = []
    store_names = re.findall(r'Store Name:\s*([^\n]+)', full_text)
    for store_name in store_names:
        store = {
            "store_name": store_name.strip(),
            "address": safe_extract(rf'{re.escape(store_name)}.*?Address:\s*([^\n]+)', full_text),
            "opening_hours": safe_extract(rf'{re.escape(store_name)}.*?Opening Hours:\s*([^\n]+)', full_text),
            "offers": safe_extract(rf'{re.escape(store_name)}.*?Offers:\s*([^\n]+)', full_text)
        }
        stores.append(store)
    return stores

def extract_inventory(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    products = []
    matches = re.finditer(r'Product:\s*([^\n]+).*?Stock Status:\s*([^\n]+)', full_text, re.DOTALL)
    for match in matches:
        product, status = match.groups()
        products.append({
            "product_name": product.strip(),
            "stock_status": status.strip()
        })
    return products

# === Ingestion ===
def ingest():
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(DATA_FOLDER, file)
        category = CATEGORY_MAP.get(file, "General")

        if category == "CustomerProfile":
            entities = extract_customer_profiles(pdf_path)
        elif category == "OrderTracking":
            entities = extract_orders(pdf_path)
        elif category == "StoreInfo":
            entities = extract_stores(pdf_path)
        elif category == "Inventory":
            entities = extract_inventory(pdf_path)
        else:
            entities = []

        for entity in entities:
            doc_obj = Document(
                page_content=entity_to_text(entity),
                metadata={**entity, "category": category, "source_file": file}
            )
            chunks = splitter.split_documents([doc_obj])
            all_docs.extend(chunks)

    if all_docs:
        vectorstore = FAISS.from_documents(all_docs, embedder)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"✅ FAISS index built with {len(all_docs)} chunks")

if __name__ == "__main__":
    ingest()
