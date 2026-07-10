"""
Learning resource builder.

Coordinates generation of all
translation outputs.
"""


from validators.content_validator import (
    validate_metadata
)



def build_resource(
    translations,
    metadata
):
    """
    Build validated learning resource.
    """


    valid, errors = validate_metadata(
        metadata
    )


    if not valid:

        raise ValueError(
            errors
        )


    resource = {

        "metadata": metadata,

        "content": translations

    }


    return resource