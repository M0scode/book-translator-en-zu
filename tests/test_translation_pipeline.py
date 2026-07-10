from pipeline.translator_pipeline import (
    translate_document
)



def test_pipeline_structure(
    tmp_path,
    monkeypatch
):

    file = tmp_path / "book.txt"

    file.write_text(
        "Education is important.",
        encoding="utf-8"
    )


    def fake_translate(text):

        return "Imfundo ibalulekile."


    monkeypatch.setattr(
        "pipeline.translator_pipeline.translate",
        fake_translate
    )


    result = translate_document(file)


    assert len(result) == 1


    assert (
        result[0]["english"]
        == "Education is important."
    )


    assert (
        result[0]["zulu"]
        == "Imfundo ibalulekile."
    )