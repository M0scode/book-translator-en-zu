"""
PDF output utilities.
"""


from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet



def save_pdf(
    resource,
    output_path,
    title="English - isiZulu Translation"
):
    """
    Generate bilingual PDF.

    Parameters:

        translations (list):
            Translation records

        output_path (str):
            PDF destination
    """
    translations = resource["content"]

    document = SimpleDocTemplate(
        output_path
    )


    styles = getSampleStyleSheet()


    content = []


    content.append(
        Paragraph(
            title,
            styles["Title"]
        )
    )


    content.append(
        Spacer(1,20)
    )


    for item in translations:


        content.append(
            Paragraph(
                f"Paragraph {item['id']}",
                styles["Heading2"]
            )
        )


        content.append(
            Paragraph(
                "<b>English:</b>",
                styles["Normal"]
            )
        )


        content.append(
            Paragraph(
                item["english"],
                styles["Normal"]
            )
        )


        content.append(
            Spacer(1,10)
        )


        content.append(
            Paragraph(
                "<b>isiZulu:</b>",
                styles["Normal"]
            )
        )


        content.append(
            Paragraph(
                item["zulu"],
                styles["Normal"]
            )
        )


        content.append(
            Spacer(1,20)
        )


    document.build(content)

    return output_path