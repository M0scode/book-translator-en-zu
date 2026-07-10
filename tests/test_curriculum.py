from models.curriculum import CurriculumMetadata



def test_curriculum_metadata():

    metadata = CurriculumMetadata(
        curriculum="CAPS",
        country="South Africa",
        grade=10,
        subject="Life Sciences",
        term=1,
        topic="Cell Structure"
    )


    assert metadata.curriculum == "CAPS"

    assert metadata.country == "South Africa"

    assert metadata.grade == 10

    assert metadata.subject == "Life Sciences"

    assert metadata.term == 1

    assert metadata.topic == "Cell Structure"