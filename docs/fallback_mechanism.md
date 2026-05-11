\# Fallback Mechanism



The project includes a rule-based fallback mechanism to ensure reliability during API failures.



\## Purpose



Large Language Models depend on external APIs that may fail due to:

\- Rate limits

\- Network issues

\- Server downtime

\- Invalid responses



The fallback mechanism ensures the system continues functioning even when the Gemini API is unavailable.



\## Fallback Techniques



\### 1. Dictionary-Based Simplification

Complex legal terms are replaced with simpler alternatives.



\### 2. Sentence Splitting

Long legal sentences are divided into shorter readable sentences.



\### 3. Extractive Summarization

Important sentences are selected directly from the original text.



\## Benefits



\- Improved system reliability

\- Continuous output generation

\- Better user experience

