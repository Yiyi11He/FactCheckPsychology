import pymupdf 

def pdf_to_json(pdf_file):
    # Open the document using pymupdf
    doc = pymupdf.open(pdf_file)
    content = []
    
    # Iterate over each page and extract the text
    for page in doc:
        content.append(page.get_text())
        
    # Close the document after processing
    doc.close()
    
    # Structure the extracted content into a dictionary and return it
    json_content = {"content": content}
    return json_content
