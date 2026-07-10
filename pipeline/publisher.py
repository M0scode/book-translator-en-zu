"""
Resource publishing coordinator.

Generates final learning resources.
"""


from outputs.json_writer import save_json

from learning.mobile_writer import (
    save_mobile_content
)



def publish_resource(
    resource,
    output_directory
):
    """
    Publish learning resources.
    """


    json_path = (
        output_directory /
        "learning_resource.json"
    )


    mobile_path = (
        output_directory /
        "mobile_learning.json"
    )


    save_json(
        resource,
        json_path
    )


    save_mobile_content(
        resource,
        mobile_path
    )


    return {
        "json": json_path,
        "mobile": mobile_path
    }