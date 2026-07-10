from outputs.pdf_writer import save_pdf

import os



def test_save_pdf(tmp_path):

    output_file = (
        tmp_path /
        "translation.pdf"
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


    save_pdf(
        translations,
        output_file
    )


    assert os.path.exists(
        output_file
    )