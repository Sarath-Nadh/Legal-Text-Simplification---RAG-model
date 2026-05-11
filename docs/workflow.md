\# System Workflow



The system follows a hybrid NLP and RAG-based workflow for legal text simplification.



\## Workflow Steps



1\. User uploads legal document

2\. Text extraction is performed

3\. NLP preprocessing cleans and normalizes text

4\. Sentence embeddings are generated

5\. FAISS retrieves relevant contextual chunks

6\. Gemini LLM simplifies legal content

7\. Readability analysis is performed

8\. Fallback mechanism activates during API failure

9\. Final simplified output is displayed

