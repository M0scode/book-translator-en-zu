from outputs.docx_writer import save_docx

from docx import Document



def test_save_docx(tmp_path):

    output_file = (
        tmp_path /
        "translation.docx"
    )


    translations = [
        {
            "id":1,
            "english":
            "Education is important.",

            "zulu":
            "Imfundo ibalulekile."
        }
    ]


    save_docx(
        translations,
        output_file
    )


    document = Document(
        output_file
    )


    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )


    assert (
        "Education is important."
        in text
    )


    assert (
        "Imfundo ibalulekile."
        in text
    )