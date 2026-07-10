from processors.text_processor import (
    clean_text,
    split_paragraphs
)



def test_clean_text():

    raw_text = """
    
    Hello world.
    
    
    This is a test.
    
    """

    cleaned = clean_text(raw_text)

    assert cleaned == (
        "Hello world.\n"
        "This is a test."
    )



def test_split_paragraphs():

    text = (
        "First paragraph.\n\n"
        "Second paragraph."
    )

    result = split_paragraphs(text)

    assert len(result) == 2

    assert result[0]["id"] == 1

    assert (
        result[0]["text"]
        == "First paragraph."
    )