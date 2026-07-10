from pipeline.resource_builder import (
    build_resource
)



def test_build_resource():

    translations = [

        {
            "id":1,
            "english":"Hello",
            "zulu":"Sawubona"
        }

    ]


    metadata = {

        "curriculum":"CAPS",

        "country":"South Africa",

        "grade":10,

        "subject":"Life Sciences",

        "term":1,

        "topic":"Cells"

    }


    resource = build_resource(
        translations,
        metadata
    )


    assert (
        resource["metadata"]["grade"]
        == 10
    )


    assert (
        resource["content"][0]["zulu"]
        == "Sawubona"
    )