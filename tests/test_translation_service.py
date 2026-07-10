from services.translation_service import (
    translate_content
)



def test_translate_text_service():

    metadata = {

        "grade": 10,

        "subject": "Life Sciences",

        "term": 1

    }


    result = translate_content(
        "Education is the key to success.",
        metadata,
        input_type="text"
    )


    assert isinstance(
        result,
        dict
    )


    assert "metadata" in result

    assert "content" in result


    assert (
        result["metadata"]["grade"]
        == 10
    )


    assert len(
        result["content"]
    ) > 0



if __name__ == "__main__":

    test_translate_text_service()

    print(
        "Translation service test passed!"
    )