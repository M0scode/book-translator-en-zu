import json

from outputs.json_writer import save_json



def test_save_json(tmp_path):

    output_file = (
        tmp_path /
        "translation.json"
    )


    translations = [
        {
            "id": 1,
            "english":
            "Education is important.",

            "zulu":
            "Imfundo ibalulekile."
        }
    ]


    metadata = {
        "subject": "Life Sciences",
        "grade": "10"
    }


    save_json(
        translations,
        output_file,
        metadata
    )


    with open(
        output_file,
        encoding="utf-8"
    ) as file:

        result = json.load(file)


    assert (
        result["metadata"]["grade"]
        == "10"
    )


    assert (
        result["content"][0]["zulu"]
        == "Imfundo ibalulekile."
    )