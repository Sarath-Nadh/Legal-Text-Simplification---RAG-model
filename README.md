Hybrid NLP and RAG-Based Legal Text Simplification System

A product-based legal document simplification system that transforms complex legal text into plain and understandable language using Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and Google Gemini AI.

📌 Project Overview

Legal documents often contain complex terminology, lengthy sentence structures, and dense legal references that are difficult for non-expert users to understand. This project addresses that challenge by providing an intelligent system capable of simplifying legal documents while preserving their original meaning and legal intent.

The system combines traditional NLP preprocessing, RAG-based contextual retrieval, Large Language Models (LLMs), and a rule-based fallback mechanism to generate simplified legal text, summaries, and readability analysis.

🚀 Features
Legal text simplification using Gemini AI
Retrieval-Augmented Generation (RAG) pipeline
Multi-format document support (TXT, PDF, Image)
Legal text preprocessing using NLP
Readability score evaluation
Legal summary generation
FAISS-based similarity search
Rule-based fallback mechanism for API failures
Interactive Streamlit user interface

🛠️ Technologies Used
Programming Language
Python
Framework
Streamlit
NLP & Processing
NLTK
Regex
TextStat
RAG Components
Sentence Transformers
FAISS Vector Database
AI / LLM
Google Gemini API
Gemini 2.5 Flash

🧠 System Workflow
User uploads legal document
Text extraction and preprocessing
NLP cleaning and normalization
Embedding generation using Sentence Transformers
Context retrieval using FAISS similarity search
Simplification using Gemini LLM
Summary and readability generation
Fallback simplification if API fails
Results displayed through Streamlit UI
