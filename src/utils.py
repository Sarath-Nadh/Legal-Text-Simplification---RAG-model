import io
import re
import os
import docx
import PyPDF2
from PIL import Image
import pytesseract
import tempfile
import requests
import numpy as np

def read_file(file_obj):
    """
    Read text from a file object based on file type.
    
    Args:
        file_obj: File object from Streamlit uploader
        
    Returns:
        str: Extracted text
    """
    # Get file extension
    file_name = file_obj.name
    file_extension = file_name.split('.')[-1].lower()
    
    # Read based on file type
    if file_extension == 'txt':
        # Read text file
        text = file_obj.getvalue().decode('utf-8')
    elif file_extension == 'pdf':
        # Read PDF file
        text = convert_pdf_to_text(file_obj)
    elif file_extension == 'docx':
        # Read DOCX file
        text = convert_docx_to_text(file_obj)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        # Read image file
        text = convert_image_to_text(file_obj)
    else:
        text = f"Unsupported file format: {file_extension}"
    
    return text

def convert_pdf_to_text(file_obj):
    """
    Extract text from a PDF file.
    
    Args:
        file_obj: PDF file object
        
    Returns:
        str: Extracted text
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_obj.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Open the PDF file
        with open(temp_file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return text

def convert_docx_to_text(file_obj):
    """
    Extract text from a DOCX file.
    
    Args:
        file_obj: DOCX file object
        
    Returns:
        str: Extracted text
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
        temp_file.write(file_obj.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Open the DOCX file
        doc = docx.Document(temp_file_path)
        
        # Extract text from all paragraphs
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        text = f"Error extracting text from DOCX: {str(e)}"
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return text

def convert_image_to_text(file_obj):
    """
    Extract text from an image using OCR.
    
    Args:
        file_obj: Image file object
        
    Returns:
        str: Extracted text
    """
    try:
        # Open the image
        image = Image.open(file_obj)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
    except Exception as e:
        text = f"Error extracting text from image: {str(e)}"
    
    return text

def clean_legal_citations(text):
    """
    Clean and standardize legal citations in the text.
    
    Args:
        text (str): Text with legal citations
        
    Returns:
        str: Text with standardized citations
    """
    # Standardize case citations
    # Example: "Smith v. Jones" -> "Smith v Jones"
    text = re.sub(r'(\w+)\s+v\.\s+(\w+)', r'\1 v \2', text)
    
    # Standardize section references
    # Example: "section 123" -> "Section 123"
    text = re.sub(r'(?i)section\s+(\d+)', r'Section \1', text)
    
    # Standardize act references
    # Example: "act, 1950" -> "Act, 1950"
    text = re.sub(r'(?i)act,\s+(\d{4})', r'Act, \1', text)
    
    return text

def segment_legal_document(text):
    """
    Segment a legal document into its constituent parts.
    
    Args:
        text (str): Legal document text
        
    Returns:
        dict: Document segments (facts, arguments, decision, etc.)
    """
    segments = {
        "header": "",
        "facts": "",
        "arguments": "",
        "legal_principles": "",
        "decision": "",
        "citations": []
    }
    
    # Simple heuristic segmentation - this would be more sophisticated in practice
    lines = text.split('\n')
    current_section = "header"
    
    for i, line in enumerate(lines):
        lower_line = line.lower()
        
        # Try to identify section transitions
        if any(keyword in lower_line for keyword in ["facts", "background", "case"]) and i > 5:
            current_section = "facts"
            continue
        elif any(keyword in lower_line for keyword in ["argument", "contention", "submitted"]):
            current_section = "arguments"
            continue
        elif any(keyword in lower_line for keyword in ["law", "principle", "rule", "statute"]):
            current_section = "legal_principles"
            continue
        elif any(keyword in lower_line for keyword in ["decision", "judgment", "order", "conclusion"]):
            current_section = "decision"
            continue
        
        # Add line to current section
        if current_section in segments:
            segments[current_section] += line + "\n"
        
        # Extract citations
        if "v." in line or "vs." in line:
            segments["citations"].append(line.strip())
    
    # Clean up segments
    for key in segments:
        if isinstance(segments[key], str):
            segments[key] = segments[key].strip()
    
    return segments

def calculate_text_complexity(text):
    """
    Calculate the complexity of legal text.
    
    Args:
        text (str): Legal text
        
    Returns:
        dict: Complexity metrics
    """
    import textstat
    
    metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "text_standard": textstat.text_standard(text)
    }
    
    return metrics