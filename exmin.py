# Implementation of the expectation min motif finding algorithm
# TODO optimaliseren in count_occurences
# TDO deftige interface makenK
from scoring import get_frequency_matrix

EPS = 0.01
BASES = 4


import random
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
    total = 0
    for i in range(BASES):
        for k in range(motif_width + 1):
            total += abs(p_old[i][k] - p_new[i][k])
    return total

#  returns a matrix with random values where the columns add up to one
def initialize_beliefs(motif_width):
    beliefs = [[] for _ in range(BASES)]

    for i in range(motif_width + 1):
        # https://stackoverflow.com/a/3590105
        dividers = sorted(random.sample(range(1, 10), BASES-1))
        values = [(a - b) / 10 for a, b in zip(dividers + [10], [0] + dividers)]
        for j in range(len(values)):
            beliefs[j].append(values[j])

    return beliefs

# calculates the probability of a sequence given a motif starting position
def prob_sequence_motif(sequence: str, motif_start: int, beliefs: list, motif_width: int):
    probability = 1
    for k in range(motif_start):
        probability *= beliefs[to_index(sequence[k])][0]
    for k in range(motif_start, motif_start+motif_width):
        probability *= beliefs[to_index(sequence[k])][k-motif_start+1]
    for k in range(motif_start+motif_width, len(sequence)):
        probability *= beliefs[to_index(sequence[k])][0]

    return probability

# expectation step of the EM algorithm (in OOPS), we guess new values for the hidden variables
def do_expectation(sequences: list, beliefs: list, motif_width: int):
    new_hidden_variables = list()
    for sequence in sequences:
        new_row = list()
        row_total = 0
        for j in range(len(sequence) - motif_width + 1):
            value = prob_sequence_motif(sequence, j, beliefs, motif_width)
            new_row.append(value)
            if j > 0: row_total += value
        # normalize
        if row_total != 0:
            new_hidden_variables.append([value/row_total for value in new_row])
        else:
            new_hidden_variables.append(new_row)
    return new_hidden_variables

# counts the amount of c's that occur at position k
def count_occurences(sequences: list, hidden_variables, motif_width, c, k):
    if k > 0:
        total = 0
        for i in range(len(sequences)):
            # sum over positions where c appears
            for j in range(len(sequences[i]) - k + 1):
                if sequences[i][j + k - 1] == c and j + k - 1 <= len(sequences[i]) - motif_width:
                    total += hidden_variables[i][j]
        return total
    else:
        total_occurences = 0
        for sequence in sequences:
            total_occurences += sequence.count(c)
        others = 0
        for j in range(1, motif_width+1):
            others += count_occurences(sequences, hidden_variables, motif_width, c, j)
        return total_occurences - others

# maximization step of the EM algorithm, we update beliefs based on our new hidden variables
def do_maximization(sequences, hidden_variables, motif_width):
    new_beliefs = list()
    for i in range(BASES):
        new_row = list()
        denominator = 0
        for k in range(motif_width + 1):
            # if count_occurences(sequences, hidden_variables, motif_width, to_char(i), k) == 0:
            #     print("oops")
            nominator = count_occurences(sequences, hidden_variables, motif_width, to_char(i), k) + 1 # plus one is a pseudocounter
            denominator += nominator
            new_row.append(nominator)

        if denominator != 0:
            new_beliefs.append([nominator/denominator for nominator in new_row])
        else:
            new_beliefs.append(new_row)
    return new_beliefs

def score_motif(sequences, starting_positions, motif):
    score = 0
    for i in range(len(starting_positions)):
        motif_index = starting_positions[i].index(max(starting_positions[i]))
        for j in range(len(motif)):
            if sequences[i][motif_index + j] == motif[j]:
                score += 1
    return score

def get_motif_from_beliefs(beliefs, motif_width):
    maximums = [0 for _ in range(motif_width)]
    motif = ['A' for _ in range(motif_width)]
    for i in range(len(beliefs)):
        for k in range(1, len(beliefs[i])):
            if beliefs[i][k] > maximums[k - 1]:
                motif[k - 1] = to_char(i)
                maximums[k - 1] = beliefs[i][k]
    str_temp = ""
    return str_temp.join(motif)

def get_motifs_from_sequences(sequences, starting_positions, motif_width, verbose = False):
    motifs = list()
    for i in range(len(starting_positions)):
        motif_index = starting_positions[i].index(max(starting_positions[i]))
        motif = sequences[i][motif_index:motif_index + motif_width]
        motifs.append(motif)
        if verbose: print(motif)
    return motifs

# iteravely runs em until a certain treshhold
def exmin(sequences, motif_width):
    old_beliefs = initialize_beliefs(motif_width)
    while True:
        hidden_variables = do_expectation(sequences, old_beliefs, motif_width)
        new_beliefs = do_maximization(sequences, hidden_variables, motif_width)
        if difference_in(old_beliefs, new_beliefs, motif_width) < EPS:
            old_beliefs = new_beliefs
        else:
            return hidden_variables, new_beliefs

def find_motif_exmin(sequences, motif_width):
    starting_positions, motif_beliefs = exmin(sequences, motif_width)
    return get_motifs_from_sequences(sequences, starting_positions, motif_width)

def best_of_exmin(sequences, motif_width, n=100):
    max_score = 0
    best_motifs = list()
    for i in range(n):
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

