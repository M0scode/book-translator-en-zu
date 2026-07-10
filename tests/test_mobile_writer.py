import json

from learning.mobile_writer import (
    save_mobile_content
)



def test_save_mobile_content(tmp_path):

    output_file = (
        tmp_path /
        "learning.json"
    )


    content = {

        "subject":
        "Life Sciences",

        "grade":
        10,

        "chapters":[]

    }


    save_mobile_content(
        content,
        output_file
    )


    with open(
        output_file,
        encoding="utf-8"
    ) as file:

        result = json.load(file)


    assert (
        result["grade"]
        == 10
    )


    assert (
        result["subject"]
        == "Life Sciences"
    )