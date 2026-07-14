"""
Mobile learning content generator.

Creates lightweight content format
for learner applications.
"""


import json



def save_mobile_resource(
    resource,
    output_path
):
    """
    Save learner-friendly content.

    Parameters:

        content (dict):
            Structured learning content

        output_path (str):
            JSON destination
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