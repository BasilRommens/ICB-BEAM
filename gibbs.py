# Implementation of the gibbs sampling algorithm
import time

from random import randint
from copy import copy
from scoring import get_scoring_matrix, score_pssm_log, get_frequency_matrix

# Set the time out constant to 1
TIME_OUT = 1


def splice_string(string, start_position, length):
    """
    :param start_position: index where the substring starts
    :param length: length of the substring
    >>> splice_string("123456789", 1, 4)
    '2345'
    """
    end_position = start_position + length
    return string[start_position:end_position]


def get_motifs(motif_positions, instances, motif_length, exclude_position=None):
    """
    Get the motif substrings of all the instances given the motif positions
    Optional: Exclude a given position
    >>> get_motifs([0, 2, 1], ["ATCGG", "GGAAA", "TCTTT"], 2, 2)
    ['AT', 'AA']
    >>> get_motifs([0, 2, 1], ["ATCGG", "GGAAA", "TCTTT"], 2)
    ['AT', 'AA', 'CT']
    """
    num_instances = len(instances)
    motifs = []
    for i in range(num_instances):
        if i != exclude_position:
            dna_string = instances[i]
            start_position = motif_positions[i]
            motif = splice_string(dna_string, start_position, motif_length)
            motifs.append(motif)
    return motifs


def get_best_position(string, scoring_matrix, motif_length):
    """
    :param scoring_matrix: Logged scorring matrix (e.g. obtained with get_scoring_matrix)
    >>> get_best_position("TTTGT", {'A': [1, 1.3], 'T': [1.3, 0.05], 'C': [1.3, 1], 'G': [0.05, 1.3]}, 2)
    3
    """
    best_position = 0
    best_score = float('inf')  # lower = better

    # Iterate over all posititions to find the best
    for i in range(len(string) - motif_length + 1):
        motif_guess = splice_string(string, i, motif_length)
        score = score_pssm_log(motif_guess, scoring_matrix)

        # Check if score is better
        if score < best_score:
            best_score = score
            best_position = i

    return best_position


def get_new_position(index, motif_positions, instances, motif_length):
    motifs = get_motifs(motif_positions, instances, motif_length, index)
    scoring_matrix = get_scoring_matrix(motifs)
    dna_string = instances[index]
    best_position = get_best_position(dna_string, scoring_matrix, motif_length)
    return best_position


def gibbs_sample(instances, motif_length):
    """
    Finds a motif that's present in all instances
    WARNING: Gibbs sampling is based on random start positions, so the result can change every time you run the code
    :param instances: List of strings, each string has the same length, each string contains the motif
    :param motif_length: The length for the motif
    :return: List of motif instances of the found motif

    # Note: This test fails sometimes, because gibbs is random based
    # >>> gibbs_sample(["CGTAC", "GTCCC", "AAGGT", "GCTGT"], 2)
    # ['GT', 'GT', 'GT', 'GT']
    """
    # Random start positions in the dna string for each instance
    motif_positions = [randint(0, len(instance) - motif_length) for instance in instances]
    # print(f"Start positions: {motif_positions}")  # for debugging

    # Bool whether the position has been changed somewhere in the algorithm
    positions_changed = True

    time_start = time.perf_counter()
    while positions_changed:
        old_positions = copy(motif_positions)

        for i in range(len(instances)):
            new_position = get_new_position(i, motif_positions, instances, motif_length)
            motif_positions[i] = new_position

        positions_changed = old_positions != motif_positions
        time_elapsed = (time.perf_counter() - time_start)
        # If the loops run longer than timeout seconds, the function will throw an exception to time out
        if time_elapsed > TIME_OUT:
            raise Exception("\033[1;93mTimed out!\033[0m")

    return get_motifs(motif_positions, instances, motif_length)


def most_occuring(item_list):
    return max(item_list, key=item_list.count)


def best_of_gibbs(instances, motif_length, num_iterations=10):
    """
    Runs gibbs_sample multiple times and returns the most occuring solution
    :param num_iterations: Times to run gibbs_sample
    # Note: Test still possible to fail because gibbs is random based, but low chance
    >>> best_of_gibbs(["CGTAC", "GTCCC", "AAGGT", "GCTGT"], 2)
    ['GT', 'GT', 'GT', 'GT']
    """
    gibs_results = []
    for _ in range(num_iterations):
        try:
            gibbs_result = gibbs_sample(instances, motif_length)
            gibs_results.append(gibbs_result)
        except Exception as e:
            print(e)
    return most_occuring(gibs_results)


if __name__ == '__main__':
    from pprint import pprint

    instances = ["CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
                 "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
                 "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
                 "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
                 "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]
    motif_instances = best_of_gibbs(instances, 8)
    pprint(motif_instances)

    frequency_matrix = get_frequency_matrix(motif_instances)
    pprint(frequency_matrix)
