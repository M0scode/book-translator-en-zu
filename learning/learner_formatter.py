"""
Learner resource formatter.

Transforms translated curriculum content
into a learner-friendly study format.
"""
from learning.glossary import GLOSSARY

def create_learner_resource(resource):
    """
    Convert translation output into
    learner study material.

    Parameters:

        resource (dict):
            Translation resource

    Returns:

        dict:
            Learner formatted resource
    """


    learner_resource = {

        "metadata": resource.get(
            "metadata",
            {}
        ),

        "sections": []

    }


    for item in resource["content"]:

        section = {

            "section_id": item["id"],

            "lesson_content": item["zulu"],

            "english_reference": item["english"],

            "key_terms": extract_key_terms(
                item["english"]
            )

        }


        learner_resource["sections"].append(
            section
        )


    return learner_resource



def extract_key_terms(text):
    """
    Placeholder glossary extraction.

    Future versions can connect
    curriculum dictionaries.
    """
    found_terms = []

    text_lower = text.lower()


    for term, details in GLOSSARY.items():

        if term in text_lower:

            found_terms.append(
                {
                    "term": term,
                    "zulu_term": details["zulu_term"],
                    "meaning": details["meaning"]
                }
            )


    return found_terms
