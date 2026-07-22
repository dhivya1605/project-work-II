import random

def choose_random(value):
    """
    Randomly selects one value from a comma-separated string.
    """

    values = [v.strip() for v in str(value).split(",")]

    return random.choice(values)