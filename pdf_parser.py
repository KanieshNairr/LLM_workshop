import fitz
import re

def extract_text_from_pdf(pdf_path_or_bytes):
    if isinstance(pdf_path_or_bytes, bytes):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)
        
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def parse_law_text(text):
    """
    Parses a combined text string of law PDF into a dictionary.
    Keys are section numbers (e.g., '103', '199A'), values are the text.
    Flattens the text to handle formatting where the Section Name and Number are separated by weird newlines.
    """
    law_dict = {}
    
    # 1. Flatten the text by removing line breaks to fix bad PDF formatting
    # Replace all newlines with a space
    flat_text = text.replace('\n', ' ')
    
    # Clean up multiple spaces
    flat_text = re.sub(r'\s+', ' ', flat_text)
    
    # 2. Section Pattern
    # This matches exactly: "199A." or "194D." or "Section 103:" 
    # Ensuring it starts after a space (or beginning) and is followed by a space and an uppercase letter or parenthesis (start of description)
    pattern = re.compile(r'(?:^|\s)(?:Section\s+)?(\d+[A-Za-z]*)\s*[\.\-\:]\s*(?=[A-Z\(])', re.IGNORECASE)
    
    # 3. Find all matches and extract the blocks between them
    matches = list(pattern.finditer(flat_text))
    
    for i in range(len(matches)):
        current_match = matches[i]
        section_num = current_match.group(1).strip()
        
        # The content starts right after the match block
        start_idx = current_match.end()
        
        # The content ends right before the NEXT match block (or end of file)
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(flat_text)
            
        section_text = flat_text[start_idx:end_idx].strip()
        
        # 4. Filter out Table of Contents (usually short strings)
        if len(section_text) > 80:
            if section_num in law_dict:
                law_dict[section_num] += "\n\n" + section_text
            else:
                law_dict[section_num] = section_text
                
    return law_dict

def process_pdf(pdf_path_or_bytes):
    text = extract_text_from_pdf(pdf_path_or_bytes)
    return parse_law_text(text)
