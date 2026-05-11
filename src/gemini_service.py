import os
import re
import io
import google.generativeai as genai
from dotenv import load_dotenv
from src.rag_service import RAGService
from PIL import Image

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found. Check .env file.")

genai.configure(api_key=api_key)

# Initialize RAG
rag = RAGService()


# -----------------------------
# Fallback Functions
# -----------------------------

def fallback_simplify(text):
    LEGAL_SIMPLIFICATIONS = {
        "appellant": "person who filed the appeal",
        "respondent": "person responding to the appeal",
        "plaintiff": "person who filed the case",
        "defendant": "person accused in the case",
        "pursuant": "according to",
        "hereinafter": "from now on",
        "aforementioned": "mentioned earlier",
        "jurisdiction": "legal authority",
        "terminate": "end",
        "commence": "start",
        "subsequent": "later",
        "prior": "before",
        "notwithstanding": "despite",
        "in accordance with": "following",
        "impugned": "challenged"
    }

    simplified = text

    for term, simple in LEGAL_SIMPLIFICATIONS.items():
        simplified = re.sub(r'\b' + term + r'\b', simple, simplified, flags=re.IGNORECASE)

    sentences = re.split(r'(?<=[.!?]) +', simplified)
    improved_sentences = []

    for sentence in sentences:
        words = sentence.split()

        if len(words) > 20:
            midpoint = len(words) // 2
            part1 = " ".join(words[:midpoint]) + "."
            part2 = " ".join(words[midpoint:])
            improved_sentences.append(part1)
            improved_sentences.append(part2)
        else:
            improved_sentences.append(sentence)

    return " ".join(improved_sentences)


def fallback_summary(text):
    sentences = text.split(".")
    summary = ". ".join(sentences[:5])
    return summary.strip()


# -----------------------------
# Gemini Core (RAG + LLM)
# -----------------------------

def get_gemini_response(task_type, text):
    """
    task_type: "simplify", "summary", "keypoints", "analysis"
    """

    try:

        if len(text) > 30000:
            text = text[:30000] + "... [text truncated]"

        # 🔥 RAG retrieval
        context = rag.retrieve(text)

        print("\n--- RAG CONTEXT ---")
        print(context)
        print("-------------------\n")

        # 🔥 Different prompts for different tasks
        if task_type == "simplify":
            instruction = """
Simplify the text into plain English for a non-legal audience.

Rules:
- Use simple words
- Avoid legal jargon
- Break long sentences
- Keep meaning accurate
"""

        elif task_type == "summary":
            instruction = """
Provide a clear and structured summary of the legal document.
"""

        elif task_type == "keypoints":
            instruction = """
Extract the most important points as a numbered list.
"""

        elif task_type == "analysis":
            instruction = """
Analyze the legal arguments and explain key reasoning and principles.
"""

        else:
            instruction = "Process the text."

        # 🔥 Final prompt
        full_prompt = f"""
You are a legal assistant.

Use the following legal context ONLY to understand the meaning.
DO NOT copy legal language.

LEGAL CONTEXT:
{context}

{instruction}

TEXT:
{text}
"""

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(full_prompt)

        return response.text

    except Exception as e:
        print("Gemini failed:", str(e))
        raise e


# -----------------------------
# Simplification
# -----------------------------

def get_gemini_simplified(text):
    try:
        return get_gemini_response("simplify", text)
    except Exception:
        print("Using fallback simplification.")
        return fallback_simplify(text)


# -----------------------------
# Summary
# -----------------------------

def get_gemini_summary(text):
    try:
        return get_gemini_response("summary", text)
    except Exception:
        print("Using fallback summary.")
        return fallback_summary(text)


# -----------------------------
# Key Points
# -----------------------------

def get_gemini_key_points(text):
    try:
        response = get_gemini_response("keypoints", text)

        points = [line.strip() for line in response.split('\n') if line.strip()]
        points = [re.sub(r'^\d+\.?\s*', '', point) for point in points]

        return points
    except Exception:
        return []


# -----------------------------
# Legal Analysis
# -----------------------------

def analyze_legal_arguments(text):
    return get_gemini_response("analysis", text)


# -----------------------------
# Gemini Vision OCR
# -----------------------------

def extract_text_from_image_gemini(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
Extract all text content from this image.
Preserve formatting and wording accurately.
"""

        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content([prompt, image])

        return response.text

    except Exception as e:
        return f"Error extracting text: {str(e)}"


# -----------------------------
# API Test
# -----------------------------

def test_api_connection():
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Test connection")
        print(response.text)
        return True
    except Exception as e:
        print("API error:", str(e))
        return False


if __name__ == "__main__":
    test_api_connection()