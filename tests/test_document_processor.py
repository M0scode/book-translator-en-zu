from processors.document_processor import (
    extract_text
)



def test_txt_extraction(tmp_path):

    file = tmp_path / "sample.txt"

    file.write_text(
        "Education is important.",
        encoding="utf-8"
    )


    result = extract_text(file)

    assert (
        result
        == "Education is important."
    )