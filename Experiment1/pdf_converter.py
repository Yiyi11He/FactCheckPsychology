import fitz

def pdf_to_json(pdf_file):
    doc = fitz.open(pdf_file)
    content = []
    for page in doc:
        content.append(page.get_text())
    doc.close()
    
    json_content = {"content": content}
    return json_content
