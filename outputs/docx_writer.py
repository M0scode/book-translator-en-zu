"""
DOCX output utilities.

Creates bilingual curriculum documents.
"""


from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH



def save_docx(
    resource,
    output_path,
    title="English - isiZulu Translation"
):
    """
    Generate a bilingual Word document.

    Parameters:

        translations (list):
            Translation records

        output_path (str):
            Destination DOCX file

        title (str):
            Document title
    """
    translations = resource["content"]

    document = Document()


    # Title

    heading = document.add_heading(
        title,
        level=1
    )

    heading.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )


    document.add_paragraph(
        "English and isiZulu curriculum translation"
    )


    document.add_page_break()



    # Content

    for item in translations:


        document.add_heading(
            f"Paragraph {item['id']}",
            level=2
        )


        document.add_paragraph(
            "English:",
            style="Intense Quote"
        )


        document.add_paragraph(
            item["english"]
        )


        document.add_paragraph(
            "isiZulu:",
            style="Intense Quote"
        )


        document.add_paragraph(
            item["zulu"]
        )


        document.add_paragraph(
            "--------------------------------"
        )


    document.save(
        output_path
    )
    
    return output_path