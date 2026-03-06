import fitz
import re

doc = fitz.open('dataset/vech law.pdf')
text = ""
for page in doc:
    text += page.get_text("text") + "\n"

with open("pdf_output_utf8.txt", "w", encoding="utf-8") as f:
    start_194d = text.find("194D. ")
    if start_194d != -1:
        f.write("--- SECTION 194D ---\n")
        f.write(text[start_194d:start_194d+500] + "\n")
        
    start_199a = text.find("199A.")
    if start_199a != -1:
        f.write("\n--- SECTION 199A ---\n")
        f.write(text[start_199a:start_199a+500] + "\n")
