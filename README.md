# 📘 DocuMentor

**AI-Powered Technical Documentation Navigator**  
Leverage LLaMA3 / DeepSeek via Ollama to understand, explore, and explain complex technical documentation.

---

## 🚀 Project Overview

**DocuMentor** is an intelligent assistant designed to help developers navigate and understand complex documentation like `scikit-learn`, `TensorFlow`, or APIs like `Stripe`.

It uses:
- 📚 Vector database (Chroma)
- 🔍 RAG (Retrieval-Augmented Generation)
- 🤖 LLM (LLaMA3/DeepSeek via Ollama)
- 💬 Natural language interface (Flask UI)

---

## 🎯 Features

- 🔎 **Ask questions** about specific functions or concepts in documentation
- 🧠 **Explain code snippets** using context-aware responses
- 📂 **Ingest docs** from Markdown, web pages, or text
- ⚡ Powered by **RAG** with LLaMA/DeepSeek via Ollama
- 🧠 Embeddings via `sentence-transformers`

---

## 🏗️ Architecture

```text
[Docs (scraped/markdown)] → [Text Splitting & Embedding] → [Chroma Vector Store]
                                                      ↓
                                               [Retriever (RAG)]
                                                      ↓
                             [Query / Code] → [LLM via Ollama] → [Answer]
