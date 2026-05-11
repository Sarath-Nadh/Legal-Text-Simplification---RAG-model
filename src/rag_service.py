from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import os

class RAGService:
    def __init__(self, data_path="data/legal_docs"):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.documents = []
        self.index = None

        # Load and index documents
        self.load_documents(data_path)

    def load_documents(self, path):
        texts = []

        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    texts.append(text)
                    self.documents.append(text)

        # 🔹 Load CSV dataset
        try:
            df = pd.read_csv("data/legal_dataset.csv")

            for row in df.iloc[:100].itertuples():   # limit for speed
                text = str(row.Response)   # ✅ YOUR COLUMN

                # simple chunking
                chunks = text.split(". ")

                for chunk in chunks:
                    if len(chunk.strip()) > 30:
                        texts.append(chunk.strip())
                        self.documents.append(chunk.strip())

        except Exception as e:
            print("CSV loading skipped:", e)

        # Convert to embeddings
        embeddings = self.model.encode(texts)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=1):
        # Convert query to embedding
        query_embedding = self.model.encode([query])

        # Search similar docs
        distances, indices = self.index.search(query_embedding, top_k)

        # Return matched documents
        results = [self.documents[i] for i in indices[0]]
        return "\n\n".join(results)