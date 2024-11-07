def extract_text_with_references(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_content = []
    
    # Iterate through pages to extract text
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        lines = page_text.split("\n")  # Split the page text into lines
        
        for line in lines:
            clean_line = line.strip()
            if len(clean_line) > 2 and not clean_line.replace(".", "").isdigit():
                text_content.append({
                    "page": page_num,
                    "text": clean_line
                })
    
    doc.close()
    return text_content


# Find the page and text reference for the fact within the extracted PDF text
def find_references(fact, pdf_text_content):
    references = []
    for content in pdf_text_content:
        if fact.lower() in content["text"].lower():
            references.append(f"Page {content['page']}: \"{content['text']}\"")
    
    return references if references else ["No specific references found."]