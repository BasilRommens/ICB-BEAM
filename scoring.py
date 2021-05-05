# Help functions for scoring a motif matrix
from copy import copy

BASES = ["A", "T", "C", "G"]


def instances_to_count_matrix(instances):
    """
    Convert known instances to count matrix (slide 17)
    :param instances: Vector of strings of the same length (containing only the letters A, T, C and G)
    :return: A dict with 4 entries (A, T, C and G), with each entry containing a list of the occurances of that letter on given position


    >>> instances_to_count_matrix(["ACC", "ATG"])
    {'A': [2, 0, 0], 'T': [0, 1, 0], 'C': [0, 1, 1], 'G': [0, 0, 1]}
    """
    motif_length = len(instances[0])
    count_matrix = {base: [0] * motif_length for base in BASES}
    for instance in instances:
        for i in range(len(instance)):
            base = instance[i]
            count_matrix[base][i] += 1
    return count_matrix


def count_to_frequency_matrix(count_matrix):
    """
    Converts a count matrix to a frequency matrix (slide 18)
    :param instances: A dict with 4 entries (A, T, C and G), with each entry containing a list of the absolute occurances of that letter on given position
    :return: A dict with 4 entries (A, T, C and G), with each entry containing a list of the relative occurances of that letter on given position

    >>> count_to_frequency_matrix({'A': [2, 0], 'T': [0, 1], 'C': [0, 1], 'G': [0, 0]})
    {'A': [1.0, 0.0], 'T': [0.0, 0.5], 'C': [0.0, 0.5], 'G': [0.0, 0.0]}
    """
    frequency_matrix = dict()
    instance_count = sum(count_matrix[base][0] for base in BASES)
    for base, occurances in count_matrix.items():
        frequency_matrix[base] = [occurance / instance_count for occurance in occurances]
    return frequency_matrix


if __name__ == '__main__':

    from pprint import pprint

    instances = ["TATTAAAA", "AATAAATA", "TACAAATA", "TTTAAGAA", "TATACATA"]
    count_matrix = instances_to_count_matrix(instances)
    frequency_matrix = count_to_frequency_matrix(count_matrix)
    pprint(count_matrix)
    pprint(frequency_matrix)
