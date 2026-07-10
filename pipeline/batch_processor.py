"""
Batch processing utilities.
"""


def create_batches(items, batch_size=10):
    """
    Split items into smaller batches.

    Parameters:
        items (list)
        batch_size (int)

    Returns:
        list of batches
    """

    batches = []

    for i in range(0, len(items), batch_size):

        batches.append(
            items[i:i + batch_size]
        )

    return batches