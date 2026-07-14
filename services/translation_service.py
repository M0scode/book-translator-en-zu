"""
Translation service layer.

Connects application interface
with translation pipeline.
"""


from pipeline.translator_pipeline import (
    translate_document,
    translate_text as pipeline_translate_text
)

from outputs.output_manager import export_outputs

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

    files = export_outputs(
    resource
)


    resource["files"] = files


    return resource