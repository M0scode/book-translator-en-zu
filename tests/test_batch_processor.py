from pipeline.batch_processor import (
    create_batches
)



def test_create_batches():

    data = [
        1,2,3,4,5,6,7
    ]


    result = create_batches(
        data,
        batch_size=3
    )


    assert len(result) == 3

    assert result[0] == [1,2,3]

    assert result[-1] == [7]