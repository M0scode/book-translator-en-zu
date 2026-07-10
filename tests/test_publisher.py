from pipeline.publisher import (
    publish_resource
)



def test_publish_resource(tmp_path):


    resource = {

        "metadata": {

            "grade":10

        },

        "content":[]

    }


    outputs = publish_resource(
        resource,
        tmp_path
    )


    assert outputs["json"].exists()

    assert outputs["mobile"].exists()