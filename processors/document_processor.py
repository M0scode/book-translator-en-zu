"""
Document extraction utilities.

Responsible for extracting text from
different document formats.
"""


from pathlib import Path

from docx import Document
from pypdf import PdfReader



def extract_text(file_path):
    """
    Extract text from supported documents.

    Supported:
        - txt
        - docx
        - pdf

    Returns:
        str
    """

    file_extension = (
        Path(file_path)
        .suffix
        .lower()
    )


    if file_extension == ".txt":
        return read_txt(file_path)

    elif file_extension == ".docx":
        return read_docx(file_path)

    elif file_extension == ".pdf":
        return read_pdf(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}"
        )



def read_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()



def read_docx(file_path):

    document = Document(file_path)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n\n".join(paragraphs)



def read_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)