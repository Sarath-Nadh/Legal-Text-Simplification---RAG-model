\# Dataset Information



The project uses a legal text dataset for Retrieval-Augmented Generation (RAG).



\## Dataset Usage



\- The dataset is stored in CSV format

\- The "Response" column is used for retrieval

\- The first 100 rows are processed for efficient development and testing

\- Text is divided into smaller chunks for improved similarity search



\## Processing Steps



1\. Load dataset using Pandas

2\. Extract relevant legal responses

3\. Perform sentence-level chunking

4\. Generate embeddings using Sentence Transformers

5\. Store embeddings in FAISS index

