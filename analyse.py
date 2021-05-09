# Compares the performance of gibbs with exmin
import resource
import time

import numpy

from gibbs import gibbs_sample, best_of_gibbs
from scoring import get_motifs_score, get_total_motifs_score, get_frequency_matrix, score_sum


def get_value(value: tuple):
    """
    This will return the value of a tuple containing a key and a value.
    :param: value: This parameter is a tuple containing a key and a value, on respectively the first and second place
    :returns: Value of an item in a dict
    """
    return value[1]


def get_min_max_motifs(motifs_score):
    """
    The function will calculate the minimal and maximal scores of all the motifs score that are created.
    :param: motifs_score: This is a dict containing per possible motif a score that is on a logarithmic scale
    :returns: The motif scores with minimum and maximum score
    """
    minimum = min(motifs_score.items(), key=get_value)
    maximum = max(motifs_score.items(), key=get_value)
    return minimum, maximum


def get_avg(motifs_score):
    """
    This function will calculate the average of all the scores in the motifs score
    dict. This is useful because we want to know what the average distance is between
    all the possible motifs. If it is a small number it is a good indicator of getting
    close to the real answer.
    :param: motifs_score: This is a dict containing per possible motif a score that is on a logarithmic scale
    :returns: The average score of all the possible motif score values
    """
    return numpy.average(list(motifs_score.values()))


def get_median(motifs_score):
    """
    This function will calculate the median of all the scores in the motifs score
    dict. This is useful because we want to know what the median distance is between
    all the possible motifs. This is a slightly better indicator about where the other
    halve is situated in the motif finding dict.
    :param: motifs_score: This is a dict containing per possible motif a score that is on a logarithmic scale
    :returns: The median score of all the possible motif score values
    """
    return numpy.median(list(motifs_score.values()))


def get_sd(motifs_score):
    """
    This function will calculate the standard deviation of all the scores in the motifs score
    dict. This is useful because we want to know how hard the different scores of motifs
    differ from each other. The lower the better, and the more evenly distributed
    :param: motifs_score: This is a dict containing per possible motif a score that is on a logarithmic scale
    :returns: The standard deviation of all the possible motif score values
    """
    return numpy.std(list(motifs_score.values()))


def get_relative_performance(_performance_dict, motifs):
    """
    This function will use the motifs that where found during
    the motif finding algorithm, and generates data about it
    :param: _performance_dict: The performance dict that needs to be filled with new
    :param: motifs: The motifs from the generated solution, will be compared with
    each other
    :returns: The dict filled with data about the performance relative to the
    motifs that were found
    """
    _performance_dict['The motifs scores'] = motifs_score = get_motifs_score(motifs)
    _performance_dict['Total of all the motifs score'] = get_total_motifs_score(motifs)
    minimum, maximum = get_min_max_motifs(motifs_score)
    _performance_dict['Minimum motif score'] = minimum
    _performance_dict['Maximum motif score'] = maximum
    _performance_dict['Average of all the motif scores'] = get_avg(motifs_score)
    _performance_dict['Standard deviation of all the motif scores'] = get_sd(motifs_score)
    _performance_dict['Median of all the motif scores'] = get_median(motifs_score)
    return _performance_dict


def get_solution_relative_performance(_performance_dict, motifs, solution):
    """
    This function will use an effective solution to generate similarity data
    on the solution given by the performance dict
    :param: _performance_dict: The performance dict that needs to be filled with new
    data
    :param: motifs: The motifs from the generated solution, will be compared
    agains a real solution
    :param: solution: This is the real motif in all the DNA-sequences
    :returns: The dict filled with data about the performance relative to the solution
    """
    # generate the matrix on which we'll perform some basic statistics
    scoring_matrix = get_frequency_matrix([solution])
    motifs_dict = dict()
    for motif in motifs:
        motifs_dict[motif] = score_sum(motif, scoring_matrix)
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m The motifs scores'] = motifs_dict
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Total of all the motifs score'] = sum(
        list(motifs_dict.values()))
    minimum, maximum = get_min_max_motifs(motifs_dict)
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Minimum motif score'] = minimum
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Maximum motif score'] = maximum
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Average of all the motif scores'] = get_avg(motifs_dict)
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Standard deviation of all the motif scores'] = get_sd(
        motifs_dict)
    _performance_dict['\033[96mOptimal:\033[0m\033[32;1m Median of all the motif scores'] = get_median(motifs_dict)
    return _performance_dict


def get_performance(solution, func, *args, **kwargs) -> dict:
    """
    Will return a bunch of statistics about the found solution. Among these statistics are: time elapsed, memory usage,
    score per motif, total score of motifs, minimum of score, maximum of score, average of all scores, standard
    deviation of scores, ... in dict form
    """
    time_start = time.perf_counter()

    motifs = func(*args, **kwargs)
    performance_dict = dict()
    # This part works for both parts described below
    performance_dict['The time elapsed (s)'] = (time.perf_counter() - time_start)
    performance_dict['Memory used (Mb)'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

    # This part is when we do not have a solution
    performance_dict = get_relative_performance(performance_dict, motifs)

    # Perform this part only when there is a solution known
    if solution is None:
        return performance_dict
    performance_dict = get_solution_relative_performance(performance_dict, motifs, solution)

    return performance_dict


def print_performance(implementation_name, performance_dict):
    """
    This prints a performance report of the performance dict that
    was passed through as a variable. The name of the implementation
    is the first argument, and the second argument must exist of a key
    with the explanation of the value connected to it.
    :param: implementation_name: The name of the implementation to whom the performance
    report belongs.
    :param: performance_dict: All the performance data corresponding to the implementation that
    was run for creating such a dict.
    :returns: nothing
    """
    print(f"\033[36;1mPerformance statistics of {implementation_name}")
    for item in performance_dict.items():
        print(f"\033[32;1m{item[0]}: \033[0m\033[3m{item[1]}\033[0m")


if __name__ == '__main__':
    instances = ["CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
                 "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
                 "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
                 "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
                 "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]
    solution = "TATAAAAA"
    length = 8
    iterations = 10000

    gibbs_performance_dict = get_performance(solution, gibbs_sample, instances, length)
    print_performance("Gibbs", gibbs_performance_dict)

    best_of_gibbs_performance_dict = get_performance(solution, best_of_gibbs, instances, length, iterations)
    print_performance("Best of gibbs", best_of_gibbs_performance_dict)
