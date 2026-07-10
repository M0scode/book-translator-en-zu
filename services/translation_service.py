"""
Translation service layer.

Connects application interface
with translation pipeline.
"""


from pipeline.translator_pipeline import (
    translate_document,
    translate_text as pipeline_translate_text
)



def translate_content(
    content,
    metadata,
    input_type="text"
):
    """
    Translate user content.

    Parameters:

        content:
            Text or file path

        metadata:
            Curriculum information

        input_type:
            text or file

    Returns:

        Learning resource
    """


    if input_type == "file":

        result = translate_document(
            content
        )

    else:

        result = pipeline_translate_text(
            content
        )


    resource = {

        "metadata": metadata,

        "content": result

    }


    return resource