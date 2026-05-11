\# System Architecture



The project follows a hybrid architecture combining NLP, Retrieval-Augmented Generation (RAG), and Large Language Models.



\## Architecture Components



\### 1. Input Layer

Accepts legal documents in TXT, PDF, and image formats.



\### 2. NLP Preprocessing Layer

Performs text cleaning, normalization, and tokenization using NLTK and Regex.



\### 3. RAG Layer

Generates embeddings using Sentence Transformers and retrieves contextual information using FAISS similarity search.



\### 4. Gemini LLM Layer

Simplifies legal text and generates summaries while preserving semantic meaning.



\### 5. Fallback Layer

Provides rule-based simplification when API failure occurs.



\### 6. Output Layer

Displays simplified text, summaries, and readability scores through Streamlit UI.

