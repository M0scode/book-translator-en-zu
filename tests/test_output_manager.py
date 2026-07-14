from outputs.output_manager import export_outputs
from pathlib import Path


def test_output_generation():

    # Sample translation resource
    resource = {

        "metadata": {

            "curriculum": "CAPS",

            "country": "South Africa",

            "grade": 10,

            "subject": "Life Sciences",

            "term": 1,

            "topic": "Photosynthesis"

        },

        "content": [

            {
                "id": 1,

                "english":
                "Photosynthesis is the process by which plants produce food using sunlight.",

                "zulu":
                "I-photosynthesis yinqubo lapho izitshalo zikhiqiza khona ukudla zisebenzisa ukukhanya kwelanga."
            },

            {
                "id": 2,

                "english":
                "Plants absorb carbon dioxide and release oxygen during this process.",

                "zulu":
                "Izitshalo zimunca i-carbon dioxide futhi zikhiphe umoya-mpilo phakathi nale nqubo."
            }

        ]

    }


    files = export_outputs(
        resource
    )


    print("\nGenerated files:")

    for file_type, path in files.items():

        print(
            file_type,
            ":",
            path
        )


        assert Path(path).exists(), (
            f"{file_type} file was not created"
        )


    print(
        "\nOutput manager test passed!"
    )


if __name__ == "__main__":

    test_output_generation()