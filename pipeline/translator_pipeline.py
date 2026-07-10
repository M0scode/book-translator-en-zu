"""
Translation pipeline.

Connects document processing,
text processing and translation.
"""


from processors.document_processor import extract_text
from processors.text_processor import (
    clean_text,
    split_paragraphs
)

from models.translator import translate

from pipeline.batch_processor import create_batches

def translate_document(file_path):
    """
    Translate a complete document.

    Parameters:
        file_path (str):
            Path to document

    Returns:
        list:
            Translation results
    """


    # 1. Extract document text

    raw_text = extract_text(file_path)


    # 2. Clean text

    cleaned_text = clean_text(raw_text)


    # 3. Split into paragraphs

    paragraphs = split_paragraphs(
        cleaned_text
    )

    # 4. Create translation batches

    batches = create_batches(
        paragraphs,
        batch_size=10
    )


    # 5. Translate each batch

    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        print(
            f"Processing batch {batch_number}/{len(batches)}"
        )


        for paragraph in batch:

            translation = translate(
                paragraph["text"]
            )


            results.append(
                {
                    "id": paragraph["id"],
                    "english": paragraph["text"],
                    "zulu": translation
                }
            )


    results = []

def translate_text(content):
    """
    Translate raw text input.

    Parameters:
        content (str):
            User provided text

    Returns:
        list:
            Translation results
    """


    # 1. Clean text

    cleaned_text = clean_text(
        content
    )


    # 2. Split paragraphs

    paragraphs = split_paragraphs(
        cleaned_text
    )


    results = []


    # 3. Translate paragraphs

    for paragraph in paragraphs:

        translation = translate(
            paragraph["text"]
        )


        results.append(
            {
                "id": paragraph["id"],

                "english":
                paragraph["text"],

                "zulu":
                translation
            }
        )
    # 4. Create translation batches

    batches = create_batches(
        paragraphs,
        batch_size=10
    )


    # 5. Translate each batch

    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        print(
            f"Processing batch {batch_number}/{len(batches)}"
        )


        for paragraph in batch:

            translation = translate(
                paragraph["text"]
            )


            results.append(
                {
                    "id": paragraph["id"],
                    "english": paragraph["text"],
                    "zulu": translation
                }
            )

    return results

