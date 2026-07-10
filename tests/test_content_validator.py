from validators.content_validator import (
    validate_metadata
)



def test_valid_metadata():

    metadata = {

        "curriculum":"CAPS",

        "country":"South Africa",

        "grade":10,

        "subject":"Life Sciences",

        "term":1,

        "topic":"Cells"

    }


    valid, errors = validate_metadata(
        metadata
    )


    assert valid is True

    assert errors == []



def test_invalid_metadata():

    metadata = {

        "subject":"Life Sciences"

    }


    valid, errors = validate_metadata(
        metadata
    )


    assert valid is False

    assert len(errors) > 0