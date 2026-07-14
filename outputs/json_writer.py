"""
JSON output utilities.

Stores translated curriculum content
in a reusable learning format.
"""


import json



def save_json(
    resource,
    output_path,
):
    """
    Save translations as JSON.

    Parameters:
        translations (list):
            Translation records

        output_path (str):
            Destination file

        
    """


    with open(
    output_path,
    "w",
    encoding="utf-8"
    ) as file:

        json.dump(
            resource,
            file,
            ensure_ascii=False,
            indent=4
        )

    return output_path