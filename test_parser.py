from pdf_parser import process_pdf
with open("dataset/vech law.pdf", "rb") as f:
    pdf_bytes = f.read()
law_dict = process_pdf(pdf_bytes)
print(f"Parsed {len(law_dict)} sections from vech law.pdf")
if len(law_dict) > 0:
    for i, (k, v) in enumerate(law_dict.items()):
        print(f"Section {k}: {v[:50]}...")
        if i > 5: break
