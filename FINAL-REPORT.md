# FINAL REPORT – Quantised LLM Capstone Project

## 1. Project Overview

This capstone project focuses on building a **Quantised LLM inference system** with:

* A FastAPI backend
* Streaming responses by default
* A simple Streamlit-based chat UI
* Clean separation of configuration, model loading, API layer, and UI

---

## 2. System Architecture

### High-Level Architecture

```
[ Streamlit UI ]
        |
        |  HTTP (Streaming)
        v
[ FastAPI Backend ]
        |
        |  OpenAI-compatible client
        v
[ Quantised LLM (Local / API) ]
```
---

## 3. Backend Design

### 3.1 Technology Stack

* **FastAPI** – API framework
* **OpenAI-compatible client** – Unified interface for local / remote models
* **Quantised LLM (Qwen 1.5B)** – Efficient inference
* **Python generators** – Token streaming
---

### 3.2 Model Serving

* The model is **served once at application startup** using `serve_model()`
* Prevents repeated loading and reduces latency
* Keeps inference layer independent from API logic

```text
Startup → Load model → Ready to accept requests
```

---

### 3.3 API Endpoints

#### `/generate`

* Stateless, single-turn inference
* Always streams output
* No server-side memory

#### `/chat`

* Stateful, multi-turn conversation
* Maintains chat history using `chat_id`

---

### 3.4 Streaming Implementation

* Streaming is **always enabled by default**
* Uses:

  * `stream=True` from the OpenAI-compatible client
  * `StreamingResponse` from FastAPI

This ensures:

* Low latency
* Progressive output
* Better UX for long responses

---

## 4. Frontend (UI) Design

### 4.1 Technology Stack

* **Streamlit** – Rapid UI development
* **requests (streaming)** – HTTP streaming support

---

### 4.2 UI Features

* Chat-style interface (`st.chat_message`)
* Sidebar controls:

  * Temperature
  * Top-p
  * Top-k
  * Max tokens
  * System prompt
* Mode switch:

  * Chat (stateful)
  * Generate (stateless)

---