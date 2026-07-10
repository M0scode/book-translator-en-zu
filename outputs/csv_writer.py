"""
CSV output utilities.
"""

import csv



def save_csv(
    translations,
    output_path
):
    """
    Save translations as CSV.

    Parameters:
        translations (list):
            Translation records

        output_path (str):
            Destination CSV file
    """


    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:


        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "english",
                "zulu"
            ]
        )


        writer.writeheader()


        writer.writerows(
            translations
        )