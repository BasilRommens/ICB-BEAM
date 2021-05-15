# Compares the performance of gibbs with exmin
import csv
import resource
import time

import numpy

from exmin import find_motif_exmin, best_of_exmin
from gibbs import gibbs_sample, best_of_gibbs
from scoring import get_motifs_score, get_total_motifs_score, \
    get_frequency_matrix, score_sum, get_motifs_percentage, \
    get_total_motifs_percentage

BASES = ["A", "T", "C", "G"]


def get_value(value: tuple):
    """
    This will return the value of a tuple containing a key and a value.
    :param: value: This parameter is a tuple containing a key and a value, on
    respectively the first and second place
    :returns: Value of an item in a dict
    """
    return value[1]


def get_min_max_motifs(motifs_score):
    """
    The function will calculate the minimal and maximal scores of all the motifs
    score that are created.
    :param: motifs_score: This is a dict containing per possible motif a score
    that is on a logarithmic scale
    :returns: The motif scores with minimum and maximum score
    """
    minimum = min(motifs_score.items(), key=get_value)
    maximum = max(motifs_score.items(), key=get_value)
    return minimum, maximum


def get_avg(motifs_score):
    """
    This function will calculate the average of all the scores in the motifs
    score dict. This is useful because we want to know what the average distance
    is between all the possible motifs. If it is a small number it is a good
    indicator of getting close to the real answer.
    :param: motifs_score: This is a dict containing per possible motif a score
    that is on a logarithmic scale
    :returns: The average score of all the possible motif score values
    """
    return numpy.average(list(motifs_score.values()))


def get_median(motifs_score):
    """
    This function will calculate the median of all the scores in the motifs
    score dict. This is useful because we want to know what the median distance
    is between all the possible motifs. This is a slightly better indicator
    about where the other halve is situated in the motif finding dict.
    :param: motifs_score: This is a dict containing per possible motif a score
    that is on a logarithmic scale
    :returns: The median score of all the possible motif score values
    """
    return numpy.median(list(motifs_score.values()))


def get_sd(motifs_score):
    """
    This function will calculate the standard deviation of all the scores in the
    motifs score dict. This is useful because we want to know how hard the
    different scores of motifs differ from each other. The lower the better, and
    the more evenly distributed
    :param: motifs_score: This is a dict containing per possible motif a score
    that is on a logarithmic scale
    :returns: The standard deviation of all the possible motif score values
    """
    return numpy.std(list(motifs_score.values()))


def get_memory_usage_mb(resource):
    """
    It will fetch the maximum memory used during the runtime of the motif finder
    :param: resource: The resource that should be checked for memory usage
    :return: The maximum memory usage in Mb
    """
    memory_usage = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    return memory_usage


def count_occurrences(instances, motifs):
    ret_dict = dict()
    for motif in motifs:
        single_count = 0
        for instance in instances:
            single_count += 1 if instance.count(motif) else 0
        ret_dict[motif] = single_count
    return ret_dict


def update_performance_dict(_performance_dict, motifs_score, total_motifs_score,
                            prefix, occurrences=None):
    _performance_dict[
        f'{prefix} The motifs scores'] = motifs_score
    _performance_dict[
        f'{prefix} Total of all the motifs score'] = total_motifs_score
    minimum, maximum = get_min_max_motifs(motifs_score)
    _performance_dict[f'{prefix} Minimum motif score'] = minimum
    _performance_dict[f'{prefix} Maximum motif score'] = maximum
    _performance_dict[f'{prefix} Average of all the motif scores'] = get_avg(
        motifs_score)
    _performance_dict[
        f'{prefix} Standard deviation of all the motif scores'] = get_sd(
        motifs_score)
    _performance_dict[f'{prefix} Median of all the motif scores'] = get_median(
        motifs_score)
    # If there is no mention of occurrences that need to be counted of the
    # motifs in the sequences then skip this step
    if occurrences is not None:
        _performance_dict[
            f'{prefix} Count of occurrences of all motifs in all strings'] = occurrences
    return _performance_dict


def get_log_relative_performance(_performance_dict, motifs):
    """
    This function will use the motifs that where found during
    the motif finding algorithm in log form, and generates data about it
    :param: _performance_dict: The performance dict that needs to be filled with
    new data
    :param: motifs: The motifs from the generated solution, will be compared
    with each other
    :returns: The dict filled with data about the performance relative to the
    motifs that were found
    """
    prefix = "\033[96mLog:\033[0m\033[32;1m"
    motifs_score = get_motifs_score(motifs)
    total_motifs_score = get_total_motifs_score(motifs)
    return update_performance_dict(_performance_dict, motifs_score,
                                   total_motifs_score, prefix)


def get_nolog_relative_performance(_performance_dict, motifs, instances):
    """
    This function will use the motifs that where found during
    the motif finding algorithm in percentage form, and generates data about it
    :param: _performance_dict: The performance dict that needs to be filled with
    new data
    :param: motifs: The motifs from the generated solution, will be compared
    with each other
    :returns: The dict filled with data about the performance relative to the
    motifs that were found
    """
    prefix = "\033[96mNolog:\033[0m\033[32;1m"
    motifs_percentage = get_motifs_percentage(motifs)
    total_motifs_percentage = get_total_motifs_percentage(motifs)
    occurrences = count_occurrences(instances, motifs)
    return update_performance_dict(_performance_dict, motifs_percentage,
                                   total_motifs_percentage, prefix, occurrences)


def get_solution_relative_performance(_performance_dict, motifs, solution):
    """
    This function will use an effective solution to generate similarity data
    on the solution given by the performance dict
    :param: _performance_dict: The performance dict that needs to be filled with
    new data
    :param: motifs: The motifs from the generated solution, will be compared
    against a real solution
    :param: solution: This is the real motif in all the DNA-sequences
    :returns: The dict filled with data about the performance relative to the
    solution
    """
    prefix = "\033[96mOptimal:\033[0m\033[32;1m"
    # generate the matrix on which we'll perform some basic statistics
    scoring_matrix = get_frequency_matrix([solution])
    motifs_dict = dict()
    # Calculate the scoring in the case of having a solution and compare to the
    # solution
    for motif in motifs:
        motifs_dict[motif] = score_sum(motif, scoring_matrix)
    total_motifs_score = sum(list(motifs_dict.values()))
    return update_performance_dict(_performance_dict, motifs_dict,
                                   total_motifs_score, prefix)


def get_general_performance(_performance_dict, time_start, resource, count):
    """
    This function will update the dict with a general inforrmation about the
    performance of the function. It reports the total runtime of the function
    and the memory usage of the function
    :param: _performance_dict: The performance dict that needs to be updated
    :param: time_start: The time that the algorithm started searching for motifs
    :param: resource: The resource that contains information about memory usage
    :returns: The performance that has nothing to do with the results of the al-
    gorithms run.
    """
    prefix = "\033[96mGeneral:\033[0m\033[32;1m"
    _performance_dict[f'{prefix} The time elapsed (s)'] = (
            time.perf_counter() - time_start)
    _performance_dict[f'{prefix} Memory used (Mb)'] = get_memory_usage_mb(
        resource)
    _performance_dict[f'{prefix} Amount of iterations'] = count
    return _performance_dict


def get_performance(solution, func, *args, **kwargs) -> dict:
    """
    Will return a bunch of statistics about the found solution. Among these
    statistics are: time elapsed, memory usage, score per motif, total score of
    motifs, minimum of score, maximum of score, average of all scores, standard
    deviation of scores, ... in dict form
    :param: solution: The expected solution from the algorithm
    :param: func: The function that needs to be benchmarked
    :returns: Performance data about the algorithms run in dict form
    """
    time_start = time.perf_counter()

    motifs, count = func(*args, **kwargs)
    performance_dict = dict()
    # This part works for both parts described below
    performance_dict = get_general_performance(performance_dict, time_start,
                                               resource, count)

    # This part is when we do not have a solution
    performance_dict = get_nolog_relative_performance(performance_dict, motifs,
                                                      instances)
    performance_dict = get_log_relative_performance(performance_dict, motifs)

    # Perform this part only when there is a solution known
    if solution is None:
        return performance_dict
    performance_dict = get_solution_relative_performance(performance_dict,
                                                         motifs, solution)

    return performance_dict


def print_performance(implementation_name, performance_dict):
    """
    This prints a performance report of the performance dict that
    was passed through as a variable. The name of the implementation
    is the first argument, and the second argument must exist of a key
    with the explanation of the value connected to it.
    :param: implementation_name: The name of the implementation to whom the
    performance report belongs.
    :param: performance_dict: All the performance data corresponding to the
    implementation that was run for creating such a dict.
    :returns: nothing
    """
    print(f"\033[36;1mPerformance statistics of {implementation_name}")
    for item in performance_dict.items():
        print(f"\033[32;1m{item[0]}: \033[0m\033[3m{item[1]}\033[0m")


def create_performance_sheet(file_name, performance_dict):
    """
    This will create a performance sheet and will append the data from the
    performance dict to a file with a file_name
    :param: file_name: The filename to which we need to write the csv
    :param: performance_dict: The data that needs to be included in the csv
    :returns: nothing
    """
    with open(file_name, 'a+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(list(performance_dict.values()))


def get_fasta_data_list(file_name) -> list:
    """
    This will create a list of all the DNA sequences in the file which should
    be in FASTA format.
    :param: file_name: The file name of the file which should contain the FASTA
    data
    :returns: The DNA sequences in string format in a list
    """
    # Opens the file
    with open(file_name) as f:
        lines = f.readlines()

    # Processes all the data in the file
    ret_list = []
    sequence = ""
    for line in lines:
        # If there is an colon in the line then it announces a new DNA string
        if ':' in line:
            continue
        sequence += line[:-1]
        # If there is an empty line with a line break
        # then there is a new DNA sequence
        if len(line) == 1:
            if not len(sequence):
                continue
            ret_list.append(sequence)
            sequence = ""
            continue
    return ret_list


def remove_unwanted_characters(strings, whitelist):
    """
    Remove all the characters that are not in the whitelist of characters from
    the strings
    :param: strings: the strings that need to be cleaned up
    :param: whitelist: the whitelist of characters allowed in the strings
    :returns: strings with only elements from the whitelist
    """
    changed = [True] * len(strings)
    # Check if one of the strings has been cleaned if not then we can return
    # otherwise we need to do another iteration
    while max(changed):
        clean_strings = []
        # This will iterate over all the strings and put the changed to false
        for i in range(len(strings)):
            string = strings[i]
            changed[i] = False
            clean_string = string
            # Iterate over every letter in the string and check for it in the
            # whitelist, if it is not in the whitelist then replace the string
            # and then leave the for loop in order to go to the next string.
            # Because it is possible we will encounter this character multiple
            # times in the string, and the original iterator over the string
            # will not be in the correct place of the string.
            for letter in string:
                if letter in whitelist:
                    continue
                clean_string = string.replace(letter, '')
                clean_strings.append(clean_string)
                changed[i] = True
                break
            if not changed[i]:
                clean_strings.append(clean_string)
        strings = clean_strings
    return strings


def make_same_length(strings):
    """
    It will make the strings that are passed through in list form of the same
    length. It does this by taking the shortest list.
    :param: strings: the strings that need to be made the same length
    :returns: strings of the same length
    """
    # Fetch the lengths of each string
    lengths = [len(string) for string in strings]
    # Take the minimal length string
    min_length = min(lengths)
    # Calculate the difference for each length of string with the lowest string
    # length
    diff_lengths = [length - min_length for length in lengths]
    # cut the last part of the string based on the string lenght difference
    strings = [
        (strings[i][:-diff_lengths[i]] if diff_lengths[i] != 0 else strings[i])
        for i in range(len(lengths))]
    return strings


def clean_up_strings(strings: list) -> list:
    """
    This function will remove the characters that are not in BASES. After this
    it will make the strings the same length. Both of these parts are necessary
    for running both implemented algorithms
    :param: strings: The strings that need to be cleaned up
    :returns: Strings of the same length and with only elements in BASES
    """
    strings = remove_unwanted_characters(strings, BASES)
    strings = make_same_length(strings)
    return strings


if __name__ == '__main__':
    temp_instances = get_fasta_data_list('testdata_16S_RNA_benoemd.FASTA')
    instances = clean_up_strings(temp_instances)
    # instances = [
    #     "CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
    #     "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
    #     "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
    #     "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
    #     "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]
    print(len(instances[0]))
    solution = None  # "TATAAAAA"
    # Variables to turn on and off running parts of the algorithm
    g = True
    bog = True
    em = True
    boem = True
    # Amount of runs
    runs = 5
    for length in range(10, 21, 10):
        for _ in range(runs):
            iterations = 50

            if g:
                gibbs_performance_dict = get_performance(solution, gibbs_sample,
                                                         instances,
                                                         length)
                print_performance("Gibbs", gibbs_performance_dict)
                create_performance_sheet('G.csv', gibbs_performance_dict)

            if bog:
                best_of_gibbs_performance_dict = get_performance(solution,
                                                                 best_of_gibbs,
                                                                 instances,
                                                                 length,
                                                                 iterations)
                print_performance("Best of gibbs",
                                  best_of_gibbs_performance_dict)
                create_performance_sheet('BOG.csv',
                                         best_of_gibbs_performance_dict)

            if em:
                em_dict = get_performance(solution, find_motif_exmin, instances,
                                          length)

                print_performance("Expectation minimization", em_dict)
                create_performance_sheet('EM.csv', em_dict)

            if boem:
                best_of_em_performance_dict = get_performance(solution,
                                                              best_of_exmin,
                                                              instances, length,
                                                              iterations)
                print_performance("Best of expectation minimization",
                                  best_of_em_performance_dict)
                create_performance_sheet('BOEM.csv',
                                         best_of_em_performance_dict)
