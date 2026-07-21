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

        resource (dict):
            Translation resource

        output_path (str):
            PDF destination
    """
    #translations = resource["content"]

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


    for section in resource["sections"]:


        content.append(
            Paragraph(
                f"Section {section['section_id']}",
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
                section["english_reference"],
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
                section["lesson_content"],
                styles["Normal"]
            )
        )


        content.append(
            Spacer(1,20)
        )
    
    if section["key_terms"]:

        content.append(
            Paragraph(
                "Key Terms",
                styles["Heading3"]
            )
        )


        for term in section["key_terms"]:

            content.append(
                Paragraph(
                    f"{term['zulu_term']} ({term['term']})",
                    styles["Normal"]
                )
            )


            content.append(
                Paragraph(
                    term["meaning"],
                    styles["Normal"]
                )
            )


    document.build(content)

    return output_path