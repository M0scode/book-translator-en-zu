"""
Learning content validation utilities.
"""


REQUIRED_METADATA_FIELDS = [

    "curriculum",
    "country",
    "grade",
    "subject",
    "term",
    "topic"

]



def validate_metadata(metadata):
    """
    Validate curriculum metadata.

    Parameters:
        metadata (dict)

    Returns:
        tuple:
            (bool, list)
    """


    errors = []


    for field in REQUIRED_METADATA_FIELDS:

        if field not in metadata:

            errors.append(
                f"Missing required field: {field}"
            )


    return (
        len(errors) == 0,
        errors
    )