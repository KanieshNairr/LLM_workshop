from pdf_parser import process_pdf
with open("dataset/vech law.pdf", "rb") as f:
    pdf_bytes = f.read()
law_dict = process_pdf(pdf_bytes)
with open("test_parser_utf8.txt", "w", encoding="utf-8") as out:
    out.write(f"Section 194D text length: {len(law_dict.get('194D', 'MISSING'))}\n")
    out.write(f"Section 194D text: {law_dict.get('194D', 'MISSING')[:200]}...\n")

    out.write(f"Section 199A text length: {len(law_dict.get('199A', 'MISSING'))}\n")
    out.write(f"Section 199A text: {law_dict.get('199A', 'MISSING')[:200]}...\n")
