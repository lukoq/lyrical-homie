# Lyrical-Homie – Polish Rap RAG Chatbot

> Chatbot that responds **only with Polish rap lyrics** – powered by Retrieval-Augmented Generation (RAG)

## Overview

**Lyrical Homie** is an experimental RAG-based chatbot that answers user queries exclusively using lines from Polish rap songs.

Instead of generating generic responses, the system:

* understands user intent,
* retrieves semantically relevant rap lyrics,
* and returns the most fitting lines — preserving **authentic style and cultural context**.

<p align="left">
      <img width="200" height="300" alt="Untitled-1" src="https://github.com/user-attachments/assets/ca92e19a-deea-4d4a-bf2d-0b13a948e36e" />
      <img width="200" height="300" alt="Untitled-2" src="https://github.com/user-attachments/assets/84e19bb1-0939-4aad-91d9-1049330af3e6" />
      <img width="200" height="300" alt="Untitled-3" src="https://github.com/user-attachments/assets/14b52572-2977-4fb7-bab9-8a7f91aaf9c8" />
</p>
---

## The approach taken

### 1. Data Collection

* Scraped 3650 songs from Genius (Polish rap artists)
* Extracted ~105k lyric lines

### 2. Preprocessing

* Language filtering using `langdetect` (Polish only)
* Lyrics split into **verses (child chunks)** with **parent context** (Parent Document Retrieval)

```json
"child": "Lorem ipsum dolor sit amet",
"parent": "Lorem ipsum dolor sit amet, consectetur adipiscing elit,",
```

---

### 3. Metadata Injection

Each verse is enriched with max 5 semantic tags using:

* `gemini-2.5-flash-lite`

Example:

```json
"tags": [
      "foo",
      "boo",
      "example"
    ]
```
Everything has been saved in the `local_metadata_registry.json` file. ([Example](lyrics_sample/example_local_metadata_registry.json))

### 4. Vector Database

* Embeddings: `sdadas/mmlw-e5-base`
* Vector DB: ChromaDB
* Similarity: cosine (`hnsw`)

---

### 5. Query Pipeline

#### Step 1: Intent Classification

Using:

* qwen2.5:7b-instruct

Classes:

* `GREETING`
* `SITUATION`
* `QUESTION`
* `SMALL_TALK`
* `SEARCH_TOPIC`

The selected intent affects the final appearance of the HYDE prompt 

---

#### Step 2: HYDE Generation

User input → transformed into a **2-line rap-style query** using:

* SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M

Prompt:

```
prompt = (
            "Jesteś polskim raperem.\n"
            f"Napisz krótką, 2-wersową odpowiedź rymowaną. {style_guide.get(intention)}\n"
            "Nie dodawaj żadnych swoich wstawek, masz wypluć sam rym.\n"
            f"User powiedział: {user_input}"
          )
```

---


#### Step 3: Retrieval Strategy

* Standard semantic search
* Metadata tag search (For `SEARCH_TOPIC`)

---

#### Step 4: Reranking
The system uses a dual-query strategy that combines the HYDE query with the user's original input

* Model: `sdadas/polish-reranker-roberta-v3`
* Top results → narrowed to **3 best candidates**

---

#### Step 5: Final Selection

* Decison maker (qwen2.5:7b-instruct) selects best match
* Extracts **top 2 most relevant lines**

---

### 6. Output

Final response:

* Always 1–2 lines of Polish rap lyrics
* Contextually aligned with user query

---

## Architecture

```
User Input
   ↓
Intent Classification (Qwen)
   ↓
HYDE (Bielik)
   ↓
Retrieval (ChromaDB + Tags)
   ↓
Reranker (RoBERTa)
   ↓
LLM Selection (Qwen)
   ↓
Final Rap Answer
```

---

## Tech Stack

* Python
* ChromaDB
* Hugging Face models:

  * `sdadas/mmlw-e5-base`
  * `sdadas/polish-reranker-roberta-v3`
* LLMs:
  
  * SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
  * qwen2.5:7b-instruct
  * gemini-2.5-flash-lite
* langdetect

---

## License

MIT

---
