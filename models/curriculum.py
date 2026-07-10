"""
Curriculum metadata model.
"""


from dataclasses import dataclass



@dataclass
class CurriculumMetadata:

    curriculum: str

    country: str

    grade: int

    subject: str

    term: int

    topic: str