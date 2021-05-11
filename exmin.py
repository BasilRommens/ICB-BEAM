# Implementation of the expectation min motif finding algorithm
from scoring import get_frequency_matrix
import random

EPS = 0.01
BASES = 4

def to_index(c):
    if c == 'A':
        return 0
    elif c == 'C':
        return 1
    elif c == 'G':
        return 2
    elif c == 'T':
        return 3

def to_char(i):
    if i == 0:
        return 'A'
    elif i == 1:
        return 'C'
    elif i == 2:
        return 'G'
    elif i == 3:
        return 'T'

def difference_in(p_old, p_new, motif_width):
    """
    calculates the absolute difference between two belief matrices
    :param p_old: the old beliefs
    :param p_new: the new beliefs
    :param motif_width: the length for the motif
    :return: the absolute difference
    """
    total = 0
    for i in range(BASES):
        for k in range(motif_width + 1):
            total += abs(p_old[i][k] - p_new[i][k])
    return total

#  returns a matrix with random values where the columns add up to one
def initialize_beliefs(motif_width):
    """
    generates a random belief matrix for a motif (in meme format)
    :param motif_width: the length for the motif
    :return: the generated belief matrix
    """
    beliefs = [[] for _ in range(BASES)]

    for i in range(motif_width + 1):
        # https://stackoverflow.com/a/3590105
        dividers = sorted(random.sample(range(1, 10), BASES-1))
        values = [(a - b) / 10 for a, b in zip(dividers + [10], [0] + dividers)]
        for j in range(len(values)):
            beliefs[j].append(values[j])

    return beliefs

def prob_sequence_motif(sequence: str, motif_start: int, beliefs: list, motif_width: int):
    """
    calculates the probability of a sequence given a motif starting position, based on a belief matrix
    :param sequence: a DNA string
    :param motif_start: given start position of the motif within the sequence
    :param beliefs: the current beliefs
    :param motif_width: the length for the motif
    :return: the generated belief matrix
    """
    probability = 1
    # probability of the sequence before the motif
    for k in range(motif_start):
        probability *= beliefs[to_index(sequence[k])][0]
    # probability of the motif
    for k in range(motif_start, motif_start+motif_width):
        probability *= beliefs[to_index(sequence[k])][k-motif_start+1]
    # probability of the sequence after the motif
    for k in range(motif_start+motif_width, len(sequence)):
        probability *= beliefs[to_index(sequence[k])][0]

    return probability

def do_expectation(sequences: list, beliefs: list, motif_width: int):
    """
    the expectation step of the EM algorithm, we calculate the expected values of hidden variables based on the belief matrix
    :param sequences: the set of dna strings
    :param beliefs: the current beliefs
    :param motif_width: the length for the motif
    :return: the guessed hidden variables
    """
    new_hidden_variables = list()
    for sequence in sequences:
        new_row = list()
        row_total = 0
        for j in range(len(sequence) - motif_width + 1):
            value = prob_sequence_motif(sequence, j, beliefs, motif_width)
            new_row.append(value)
            if j > 0: row_total += value
        # normalize, we assume that it is equally likely that the motif will start in any position
        if row_total != 0:
            new_hidden_variables.append([value/row_total for value in new_row])
        else:
            new_hidden_variables.append(new_row)
    return new_hidden_variables

def count_occurences(sequences: list, hidden_variables, motif_width, c, k):
    """
    helper function to calculate # of c’s at position k in all sequences
    :param sequences: the set of dna strings
    :param hidden_variables: hidden variables
    :param motif_width: the length for the motif
    :param c: the character we need to count
    :param k: the position in the sequence
    :return: # of c’s at position k in all sequences
    """
    if k > 0:
        total = 0
        for i in range(len(sequences)):
            # sum over positions where c appears
            for j in range(len(sequences[i]) - k + 1):
                if sequences[i][j + k - 1] == c and j + k - 1 <= len(sequences[i]) - motif_width:
                    # add the probability
                    total += hidden_variables[i][j]
        return total
    else:
        # column 0 in the belief matrix represent the background
        total_occurences = 0
        # find total # c's in the entire data set
        for sequence in sequences:
            total_occurences += sequence.count(c)
        others = 0
        for j in range(1, motif_width+1):
            others += count_occurences(sequences, hidden_variables, motif_width, c, j)
        return total_occurences - others

def do_maximization(sequences, hidden_variables, motif_width):
    """
    maximization step of the EM algorithm, create new beliefs based on the hidden variables
    :param sequences: the set of dna strings
    :param hidden_variables: hidden variables
    :param motif_width: the length for the motif
    :return: new beliefs
    """
    new_beliefs = list()
    for i in range(BASES):
        new_row = list()
        denominator = 0
        for k in range(motif_width + 1):
            nominator = count_occurences(sequences, hidden_variables, motif_width, to_char(i), k) + 1 # plus one is a pseudocounter
            denominator += nominator
            new_row.append(nominator)
        # denominator can never be zero because of the psuedocounters
        new_beliefs.append([nominator/denominator for nominator in new_row])
    return new_beliefs

def score_motif(sequences, starting_positions, motif):
    """
    simple scoring metric, looks at the starting positions with the highest chance and checks how many characters fit the motif given
    :param sequences: the set of dna strings
    :param starting_positions: matrix with probabilities for the starting position of the motif
    :param motif_width: the length for the motif
    :return: score calculated
    """
    score = 0
    for i in range(len(starting_positions)):
        # the index of the motif is the highest probability for that sequence
        motif_index = starting_positions[i].index(max(starting_positions[i]))
        for j in range(len(motif)):
            if sequences[i][motif_index + j] == motif[j]:
                score += 1
    return score

def get_motif_from_beliefs(beliefs, motif_width):
    """
    returns the most likely motif from a belief matrix, aka for every row the most likely character
    :param beliefs: belief matrix
    :param motif_width: the length for the motif
    :return: the motif found
    """
    maximums = [0 for _ in range(motif_width)]
    motif = ['A' for _ in range(motif_width)]
    for i in range(len(beliefs)):
        for k in range(1, len(beliefs[i])):
            # for every column we need to find the maximum
            if beliefs[i][k] > maximums[k - 1]:
                motif[k - 1] = to_char(i)
                maximums[k - 1] = beliefs[i][k]
    str_temp = ""
    return str_temp.join(motif)

def get_motifs_from_sequences(sequences, starting_positions, motif_width, verbose = False):
    """
    finds the motifs in the sequences based on the starting position matrix
    :param motif_width: the length for the motif
    :param starting_positions: matrix with probabilities for the starting position of the motif
    :return: list with the found motif per sequence
    """
    motifs = list()
    for i in range(len(starting_positions)):
        motif_index = starting_positions[i].index(max(starting_positions[i]))
        motif = sequences[i][motif_index:motif_index + motif_width]
        motifs.append(motif)
        if verbose: print(motif)
    return motifs

def exmin(sequences, motif_width):
    """
    run the expectation minimization algorithm until the change in beliefs is smaller than EPS
    :param sequences: the set of dna strings
    :param motif_width: the length for the motif
    :return: the probabilities of the hidden variables and the belief matrix
    """
    old_beliefs = initialize_beliefs(motif_width)
    while True:
        hidden_variables = do_expectation(sequences, old_beliefs, motif_width)
        new_beliefs = do_maximization(sequences, hidden_variables, motif_width)
        if difference_in(old_beliefs, new_beliefs, motif_width) < EPS:
            old_beliefs = new_beliefs
        else:
            return hidden_variables, new_beliefs

def find_motif_exmin(sequences, motif_width):
    """
    runs the EM algorithm one time
    :param sequences: the set of dna strings
    :param motif_width: the length for the motif
    :return: list of the motifs found by EM
    """
    starting_positions, motif_beliefs = exmin(sequences, motif_width)
    return get_motifs_from_sequences(sequences, starting_positions, motif_width)

def best_of_exmin(sequences, motif_width, iterations=100):
    """
    runs the EM algorithm multiple times and returns the best result, since EM is random
    :param sequences: the set of dna strings
    :param motif_width: the length for the motif
    :param iterations: amount of iterations to run
    :return: best list of the motifs
    """
    max_score = 0
    best_motifs = list()
    for _ in range(iterations):
        starting_positions, motif_beliefs = exmin(sequences, motif_width)
        found_motifs = get_motifs_from_sequences(sequences, starting_positions, motif_width)
        most_likely_motif = get_motif_from_beliefs(motif_beliefs, motif_width)
        score = score_motif(sequences, starting_positions, most_likely_motif)
        if score > max_score:
            max_score = score
            best_motifs = found_motifs
    return best_motifs

if __name__ == '__main__':
    from pprint import pprint

    sequences = ["CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
                 "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
                 "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
                 "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
                 "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]

    # print(best_of_exmin(test, 8, 10))
    motifs = best_of_exmin(sequences, 8)
    pprint(motifs)
    pprint(get_frequency_matrix(motifs))


