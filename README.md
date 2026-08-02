# IronStore AI Assistant

An internal RAG-based chatbot that lets employees ask natural-language questions and get answers grounded in company PDFs (policies, procedures, department docs), with source citations.

## Files

| File | Purpose |
|---|---|
| `01_ingest_and_index.ipynb` | Parses PDFs and upserts to Pinecone |
| `02_retrieve_and_answer.ipynb` | Retrieval + LLM answer generation (RAG pipeline)|
| `app.py` | Streamlit chat UI |

## Setup

1. **Document structure** — place PDFs here:
   ```
   documents/departments/<department_name>/*.pdf
   ```
   e.g. `documents/internal_docs_by_area/hr/leave_policy.pdf`


2. **Create a `.env` file** in the project root:
   ```
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=pcsk-...
   ```

## Usage

   ```bash
   streamlit run app.py
   ```

## Notes

- Uses OpenAI `text-embedding-3-large` for embeddings and `gpt-4o-mini` for answers.
- Chunks are organized by document section/heading (not fixed page size), with
  metadata (`department`, `document_name`, `section_title`, page range) attached
  to every chunk for accurate source citations.
