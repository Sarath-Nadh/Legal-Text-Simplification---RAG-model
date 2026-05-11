import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import PyPDF2
from textstat import flesch_kincaid_grade

# Import required modules
from src.text_processor import preprocess_text
from src.utils import read_file
from src.gemini_service import get_gemini_simplified, get_gemini_summary

# Load environment variables
load_dotenv()

# Configure Tesseract path if specified
tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Page configuration
st.set_page_config(
    page_title="Legal Text Simplification ",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("Legal Text Simplification")
st.write(
    "This shows document ingestion, preprocessing, simplification, summarization, "
    "and readability evaluation of legal text."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a legal document (.txt, .pdf, image)",
    type=["txt", "pdf", "jpg", "jpeg", "png"]
)

# Helper: extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Helper: extract text from image
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"Error extracting image text: {str(e)}"

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        file_type = uploaded_file.name.split('.')[-1].lower()

        # Raw text extraction
        if file_type == "pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif file_type in ["jpg", "jpeg", "png"]:
            raw_text = extract_text_from_image(uploaded_file)
        else:
            raw_text = read_file(uploaded_file)

    # Display raw text
    st.subheader("Raw Extracted Legal Text")
    st.text_area("Raw Text", raw_text, height=200)

    # Preprocessing
    preprocessed_text = preprocess_text(raw_text)

    st.subheader("Preprocessed Legal Text")
    st.text_area("Preprocessed Text", preprocessed_text, height=200)

    # Simplification
    st.subheader("Simplified Legal Text")
    with st.spinner("Simplifying legal text..."):
        simplified_text = get_gemini_simplified(preprocessed_text)
        st.text_area("Simplified Text", simplified_text, height=200)

    # Readability Comparison
    st.subheader("Readability Comparison")

    try:
        before_score = flesch_kincaid_grade(raw_text)
        after_score = flesch_kincaid_grade(simplified_text)

        col1, col2 = st.columns(2)

        col1.metric("Before Simplification (Grade Level)", round(before_score, 2))
        col2.metric("After Simplification (Grade Level)", round(after_score, 2))

        if after_score < before_score:
            st.success("✔ Readability Improved After Simplification")
        else:
            st.warning("⚠ Readability Did Not Improve Significantly")

    except Exception as e:
        st.error(f"Readability calculation failed: {e}")

    # Summarization
    st.subheader("Document Summary")
    with st.spinner("Generating summary..."):
        summary_text = get_gemini_summary(preprocessed_text)
        st.text_area("Summary", summary_text, height=150)

    st.success("Simplification, summarization, and evaluation completed successfully.")

else:
    st.info("👆 Upload a legal document to proceed.")