"""
Text processing utilities.

Responsible for cleaning and preparing
text before translation.
"""


def clean_text(text):
    """
    Remove unnecessary whitespace.

    Parameters:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """

    text = text.strip()

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return "\n".join(lines)



def split_paragraphs(text):
    """
    Split text into paragraphs.

    Parameters:
        text (str): Cleaned text

    Returns:
        list: Paragraph dictionaries
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


    return [
        {
            "id": index + 1,
            "text": paragraph
        }
        for index, paragraph in enumerate(paragraphs)
    ]