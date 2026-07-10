"""
JSON output utilities.

Stores translated curriculum content
in a reusable learning format.
"""


import json



def save_json(
    translations,
    output_path,
    metadata=None
):
    """
    Save translations as JSON.

    Parameters:
        translations (list):
            Translation records

        output_path (str):
            Destination file

        metadata (dict):
            Optional book information
    """


    output = {
        "metadata": metadata or {},
        "content": translations
    }


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )