"""
Coordinates generation of translation outputs.
"""


from pathlib import Path

from outputs.docx_writer import save_docx
from outputs.pdf_writer import save_pdf
from outputs.csv_writer import save_csv
from outputs.json_writer import save_json
from learning.mobile_writer import (
    save_mobile_resource
)

from learning.learner_formatter import (
    create_learner_resource
)


OUTPUT_DIR = Path(
    "generated"
)


def export_outputs(resource):
    """
    Generate all export formats.

    Parameters:
        resource (dict):
            Translation resource

    Returns:
        dict:
            Generated file paths
    """
    learner_resource = create_learner_resource(
        resource
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    files = {}


    files["docx"] = save_docx(
    learner_resource,
    str(OUTPUT_DIR / "translation.docx")
    )


    files["pdf"] = save_pdf(
    learner_resource,
    str(OUTPUT_DIR / "translation.pdf")
    )


    files["csv"] = save_csv(
    resource,
    str(OUTPUT_DIR / "translation.csv")
    )



    files["json"] = save_json(
        learner_resource,
        str(OUTPUT_DIR / "translation.json")
    )


    files["mobile"] = save_mobile_resource(
        resource,
        str(OUTPUT_DIR / "learner_resource.json")
    )


    return files