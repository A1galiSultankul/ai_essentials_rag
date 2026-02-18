# RAG Tutorial – How to Run

This guide explains how to run the full pipeline: **populate Qdrant from PDFs (Ollama embeddings)**, **run the Gemini-based agent with a retrieval tool**, and **evaluate with LLM-as-judge** on the project’s question-answer pairs.

---

## 1. Why you always saw `chunk_index: 0`

The code was using `pdf2chunks()` from `processing.py`, which returns a **dictionary** `{filename: [chunk0, chunk1, ...]}`, not a list. The loop was:

```python
chunks = pdf2chunks(pdf_file, ...)   # dict with one key: filename
for idx, chunk in enumerate(chunks):  # iterating over dict = iterating over keys
```

So:
- The only “chunk” was the **filename** (e.g. `"2412.15235v1.pdf"`).
- `idx` was always `0` (one key).
- `chunk_size: 16` was the length of that filename string.

**Fix applied:** we now take the list of text chunks for that file:

```python
chunks_dict = pdf2chunks(pdf_file, ...)
chunks = chunks_dict.get(pdf_file.name, [])
```

After **re-populating Qdrant** (see below), you should see real text in the dashboard and correct `chunk_index` values (0, 1, 2, …) and `chunk_size` per chunk.

---

## 2. Prerequisites

- **Docker** – for Qdrant
- **Python 3.9+** – for scripts
- **Ollama** – for embeddings (e.g. `nomic-embed-text`)
- **Gemini API key** – free at [Google AI Studio](https://aistudio.google.com/apikey)

---

## 3. Step-by-step

### 3.1 Start Qdrant

From the project root:

```bash
docker compose up -d
```

Check the dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 3.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Ollama embedding model

Install and run Ollama, then pull the embedding model:

```bash
ollama pull nomic-embed-text
ollama serve   # if not already running (often runs in background)
```

The code uses `nomic-embed-text` (768 dimensions). If you use another model, change `embedding_manager` / `populate_qdrant` and ensure the Qdrant collection vector size matches.

### 3.4 (Optional) Delete old collection and re-create

If you already had a collection filled with the buggy data (filename-as-chunk), delete it so we can re-create it with real chunks.

Using Qdrant REST API:

```bash
curl -X DELETE 'http://localhost:6333/collections/pdf_documents'
```

Or in the Qdrant dashboard: delete the `pdf_documents` collection.

### 3.5 Populate Qdrant with PDF chunks (Ollama embeddings)

From the project root:

```bash
python -m services.populate_qdrant
```

Or:

```bash
cd services && python populate_qdrant.py && cd ..
```

This will:
- Read all PDFs in `data/`
- Chunk them (size 1000, overlap 200)
- Embed each chunk with Ollama `nomic-embed-text`
- Upsert into the `pdf_documents` collection with payload: `text`, `source_file`, `chunk_index`, `chunk_size`

After this, in the Qdrant dashboard you should see real chunk text and `chunk_index` 0, 1, 2, … per file.

### 3.6 Set Gemini API key

```bash
export GEMINI_API_KEY="your-key-from-ai-studio"
```

(Or set it in your shell profile / `.env` and source it.)

### 3.7 Run the agent (retrieval tool + Gemini)

The agent:
1. **Retrieves** relevant chunks from Qdrant (the “tool”).
2. **Generates** an answer with Gemini using that context.

One-off question:

```bash
cd services && python -c "
from agent import answer_with_rag
out = answer_with_rag('What is RAG and why was it introduced?')
print('Answer:', out['answer'])
print('Sources:', out['sources'])
"
```

Or use the helper that returns only the answer string:

```bash
cd services && python -c "
from agent import answer_with_agent_tool
print(answer_with_agent_tool('What is RAG?'))
"
```

### 3.8 QA pairs for evaluation

The evaluation script expects a CSV with at least **question** and **answer** columns (e.g. `id`, `paper`, `question`, `answer`).

- **Option A – Export from the project Google Sheet**  
  - Open: [RAG Documents – Google Sheet](https://docs.google.com/spreadsheets/d/1r61qoHTzW2Zc-c60Wiao0SM8vVECN-XLR1y8ChBkLG0/edit?usp=sharing)  
  - File → Download → Comma-separated values (.csv)  
  - Save as `data/qa_pairs.csv` in the project.

- **Option B – Use the sample file**  
  - A small sample is in `data/qa_pairs_sample.csv`.  
  - Run evaluation with:  
    `--qa-csv data/qa_pairs_sample.csv`

### 3.9 Run LLM-as-judge evaluation

From the project root:

```bash
# Full QA set (after you export from the sheet as data/qa_pairs.csv)
python evaluate.py --qa-csv data/qa_pairs.csv --output-csv results/evaluation_results.csv

# Or quick test on sample (first N rows)
python evaluate.py --qa-csv data/qa_pairs_sample.csv --output-csv results/sample_results.csv
```

Optional arguments:
- `--max-samples N` – limit to first N pairs (for quick runs).
- `--judge-model gemini-1.5-flash` – model used as judge (and for RAG in the script).
- `--rag-top-k 5` – number of chunks retrieved per question.
- `--rag-threshold 0.3` – minimum similarity score for retrieved chunks.

The script will:
1. Load QA pairs from the CSV.
2. For each question: run the RAG agent (retrieval + Gemini generation).
3. Use Gemini as judge: compare model answer vs reference answer (YES/NO + short reason).
4. Report **accuracy** (fraction of judge YES) and optionally write per-row results to `--output-csv`.

Create the output directory if needed:

```bash
mkdir -p results
```

---

## 4. Summary of what’s implemented

| Requirement | Implementation |
|-------------|----------------|
| Populate Qdrant with chunks from PDFs | `services/populate_qdrant.py` – reads `data/*.pdf`, chunks with overlap, embeds via Ollama `nomic-embed-text`, upserts to `pdf_documents`. |
| Ollama-based embedding model | `services/embedding_manager.py` – calls Ollama API (`nomic-embed-text`). |
| Agent with retrieval tool | `services/agent.py` – retrieval tool in `services/retrieval_tool.py` (search Qdrant); agent uses it to get context then calls Gemini to generate the answer. |
| Free Gemini model | Uses `gemini-1.5-flash` (configurable); set `GEMINI_API_KEY` from Google AI Studio. |
| LLM-as-judge assessment | `evaluate.py` – for each QA pair runs RAG, then Gemini judge (same key facts as reference? YES/NO), reports accuracy on the collected QA pairs. |
| Question-answer pairs | CSV exported from the project [Google Sheet](https://docs.google.com/spreadsheets/d/1r61qoHTzW2Zc-c60Wiao0SM8vVECN-XLR1y8ChBkLG0/edit?usp=sharing); columns: id, paper, question, answer. |

---

## 5. Quick reference commands

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Install deps + Ollama embedding model
pip install -r requirements.txt
ollama pull nomic-embed-text

# 3. (Optional) Remove old collection
curl -X DELETE 'http://localhost:6333/collections/pdf_documents'

# 4. Populate Qdrant
python -m services.populate_qdrant

# 5. Set Gemini key and run evaluation
export GEMINI_API_KEY="..."
mkdir -p results
python evaluate.py --qa-csv data/qa_pairs.csv --output-csv results/evaluation_results.csv
```

After re-populating, the Qdrant dashboard should show real chunks and correct `chunk_index` values.
