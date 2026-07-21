from learning.learner_formatter import create_learner_resource


def test_formatter():

    resource = {

        "metadata": {
            "grade":10,
            "subject":"Physical Sciences"
        },

        "content":[
            {
                "id":1,
                "english":"Photosynthesis is the process...",
                "zulu":"I-Photosynthesis inqubo..."
            }
        ]

    }


    result = create_learner_resource(
        resource
    )


    print(result)


if __name__ == "__main__":
    test_formatter()