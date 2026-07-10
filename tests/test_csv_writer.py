import csv

from outputs.csv_writer import save_csv



def test_save_csv(tmp_path):

    output_file = (
        tmp_path /
        "translation.csv"
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


    save_csv(
        translations,
        output_file
    )


    with open(
        output_file,
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)


    assert len(rows) == 1


    assert (
        rows[0]["zulu"]
        == "Imfundo ibalulekile."
    )