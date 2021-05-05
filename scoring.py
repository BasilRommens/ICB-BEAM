# Help functions for scoring a motif matrix
from copy import copy
from math import log

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


def score_pssm(string, scoring_matrix):
    """
    Scores a DNA string using given scoring_matrix (slide 19)
    Motif length of scoring matrix must be the same as the string length
    :param string: DNA string, consist only of letters A, T, C or G
    :param scoring_matrix: Dict containing for every base (A, T, C and G) a list with the relative frequency of the base in a position
    :return: Rating between 0 (lowest match) and 1 (highest match)

    >>> score_pssm("AT", {'A': [1.0, 0.0], 'T': [0.0, 0.5], 'C': [0.0, 0.5], 'G': [0.0, 0.0]})
    0.5
    """
    score = 1
    for i in range(len(string)):
        base = string[i]
        score *= scoring_matrix[base][i]
    return score


def score_pssm_log(string, log_matrix):
    """
    Scores a DNA string using given scoring_matrix which is logged (slide 21)
    Lower score is better
    """
    score = 0
    for i in range(len(string)):
        base = string[i]
        score += log_matrix[base][i]
    return score


def add_pseudo_counts(frequency_matrix, low_frequency=0.1):
    """
    Replaces all zeros with a low frequency
    >>> add_pseudo_counts({'A': [1.0, 0.0], 'T': [0.0, 0.5], 'C': [0.0, 0.5], 'G': [0.0, 0.0]})
    {'A': [1.0, 0.1], 'T': [0.1, 0.5], 'C': [0.1, 0.5], 'G': [0.1, 0.1]}
    """
    pseudo_matrix = dict()
    for base, occurances in frequency_matrix.items():
        # TODO: Substract from other numbers, so sum for each position is still 1
        # zero_count = sum(map(lambda i: i == 0, occurances))
        # nonzero_count = len(BASES)-zero_count
        # diff = (zero_count * low_frequency) / nonzero_count
        # pseudo_matrix[base] = [low_frequency if i == 0 else i for i in occurances]
        pseudo_matrix[base] = [low_frequency if i == 0 else i for i in occurances]
    return pseudo_matrix


def freq_to_log_matrix(frequency_matrix):
    """
    Takes the log of all elements in the matrix
    Note: Pseudocounts must be added!!! (else log 0 -> error)
    """
    log_matrix = dict()
    for base, occurances in frequency_matrix.items():
        log_matrix[base] = [-log(i) for i in occurances]
    return log_matrix


def get_scoring_matrix(instances):
    count_matrix = instances_to_count_matrix(instances)
    frequency_matrix = count_to_frequency_matrix(count_matrix)
    pseudo_matrix = add_pseudo_counts(frequency_matrix)
    log_matrix = freq_to_log_matrix(pseudo_matrix)
    return log_matrix


if __name__ == '__main__':
    from pprint import pprint

    print("Count Matrix")
    instances = ["TATTAAAA", "AATAAATA", "TACAAATA", "TTTAAGAA", "TATACATA"]
    count_matrix = instances_to_count_matrix(instances)
    pprint(count_matrix)

    print("Frequency Matrix")
    frequency_matrix = count_to_frequency_matrix(count_matrix)
    pprint(frequency_matrix)

    print("Scoring PSSM")
    score1 = score_pssm("TATACATA", frequency_matrix)
    score2 = score_pssm("GCATCATT", frequency_matrix)
    score3 = score_pssm("TATAAAAT", frequency_matrix)
    print(f"scores: {score1}, {score2} and {score3}")

    print("Pseudocounts")
    pseudo_matrix = add_pseudo_counts(frequency_matrix)
    pprint(pseudo_matrix)

    print("Log Matrix")
    log_matrix = freq_to_log_matrix(pseudo_matrix)
    pprint(log_matrix)
    score1 = score_pssm_log("TATACATA", log_matrix)
    score2 = score_pssm_log("GCATCATT", log_matrix)
    score3 = score_pssm_log("TATAAAAT", log_matrix)
    print(f"scores: {score1}, {score2} and {score3}")
