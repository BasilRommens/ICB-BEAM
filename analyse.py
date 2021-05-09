# Compares the performance of gibbs with exmin
import time
import resource
import numpy

from gibbs import gibbs_sample, best_of_gibbs
from scoring import get_motifs_score, get_total_motifs_score


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


def get_performance(solution, func, *args, **kwargs):
    """
    Will return a bunch of statistics about the found solution. Among these statistics are: time elapsed, memory usage,
    score per motif, total score of motifs, minimum of score, maximum of score, average of all scores, standard
    deviation of scores, ...
    """
    time_start = time.perf_counter()

    motifs = func(*args, **kwargs)
    # This part works for both parts described below
    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

    # This part is when we do not have a solution
    motifs_score = get_motifs_score(motifs)
    total_motifs_score = get_total_motifs_score(motifs)
    minimum, maximum = get_min_max_motifs(motifs_score)
    avg = get_avg(motifs_score)
    standard_deviation = get_sd(motifs_score)
    median = get_median(motifs_score)
    # TODO This part is when we have a solution

    return time_elapsed, memMb, motifs_score, total_motifs_score, minimum, maximum, avg, standard_deviation, median


def print_performance(implementation_name, time, memory, motifs_score, total_motifs_score, minimum, maximum, avg,
                      standard_deviation, median):
    print(f"\033[32;1mOur \'{implementation_name}\' implementation:\033[0m\t\t\t {time}\t {memory}")
    print(f"\033[32;1mThe motifs scores: \033[0m{motifs_score}")
    print(f"\033[32;1mTotal of all the motifs score: \033[0m{total_motifs_score}")
    print(f"\033[32;1mMinimum motif score: \033[0m{minimum}")
    print(f"\033[32;1mMaximum motif score: \033[0m{maximum}")
    print(f"\033[32;1mAverage of all the motif scores: \033[0m{avg}")
    print(f"\033[32;1mStandard deviation of all the motif scores: \033[0m{standard_deviation}")
    print(f"\033[32;1mMedian of all the motif scores: \033[0m{median}")


if __name__ == '__main__':
    print(f"\033[36;1mPerformance statistics\t\t\t\t Time (s)\t\t\t\t Memory (Mb)\033[0m")

    instances = ["CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
                 "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
                 "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
                 "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
                 "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]
    solution = "TATAAAAA"
    length = 8
    iterations = 10000

    gibbs_time, gibbs_memory, motifs_score, total_motifs_score, minimum, maximum, avg, standard_deviation, median = get_performance(
        solution, gibbs_sample, instances, length)
    print_performance("Gibbs", gibbs_time, gibbs_memory, motifs_score, total_motifs_score, minimum, maximum, avg,
                      standard_deviation, median)

    best_gibbs_time, best_gibbs_memory, best_motifs_score, best_total_motifs_score, best_minimum, best_maximum, best_avg, best_standard_deviation, best_median = get_performance(
        solution, best_of_gibbs, instances, length, iterations)
    print_performance("Best of gibbs", best_gibbs_time, best_gibbs_memory, best_motifs_score, best_total_motifs_score,
                      best_minimum, best_maximum, best_avg, best_standard_deviation, best_median)
