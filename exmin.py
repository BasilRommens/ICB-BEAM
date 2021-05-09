# Implementation of the expectation min motif finding algorithm
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
    for i in range(4):
        for k in range(motif_width + 1):
            total += abs(p_old[i][k] - p_new[i][k])
    return total

#  returns a matrix with random values where the columns add up to one
def initialize_beliefs(motif_width):
    amount_of_bases = 4
    beliefs = [[] for _ in range(amount_of_bases)]

    for i in range(motif_width + 1):
        # https://stackoverflow.com/a/3590105
        dividers = sorted(random.sample(range(1, 10), amount_of_bases-1))
        values = [(a - b) / 10 for a, b in zip(dividers + [10], [0] + dividers)]
        for j in range(len(values)):
            beliefs[j].append(values[j])

    return beliefs

# calculates the probability of a sequence given a motif starting position
def prob_sequence_motif(sequence: str, motif_start: int, beliefs: list, motif_width: int):
    before_motif = 1
    for k in range(motif_start):
        before_motif *= beliefs[to_index(sequence[k])][0]
    motif = 1
    for k in range(motif_start, motif_start+motif_width):
        motif *= beliefs[to_index(sequence[k])][k-motif_start+1]
    after_motif = 1
    for k in range(motif_start+motif_width, len(sequence)):
        before_motif *= beliefs[to_index(sequence[k])][0]

    # assume that the starting position is uniformly distributed over all positions
    return before_motif*motif*after_motif

# expectation step of the EM algorithm (in OOPS), we guess new values for the hidden variables
def do_expectation(sequences: list, beliefs: list, motif_width: int):
    new_hidden_variables = list()
    for sequence in sequences:
        values = list()
        row_total = 0
        for j in range(len(sequence) - motif_width + 1):
            value = prob_sequence_motif(sequence, j, beliefs, motif_width)
            values.append(value)
            row_total += value
        # normalize values
        new_hidden_variables.append([value/row_total for value in values])
    return new_hidden_variables

# counts the amount of c's that occur at position k
def count_occurences(sequences: list, hidden_variables, motif_width, c, k):
    if k > 0:
        total = 0
        for i in range(len(sequences)):
            # sum over positions where c appears
            for j in range(len(sequences[i]) - k + 1):
                if sequences[i][j] == c and j + k - 1 < len(sequences[i]) - motif_width:
                    total += hidden_variables[i][j]
        return total
    else:
        total_occurences = 0
        for sequence in sequences:
            total_occurences += sequence.count(c)
        others = 0
        for j in range(1, motif_width):
            others += count_occurences(sequences, hidden_variables, motif_width, c, j)
        return total_occurences - others

# maximization step of the EM algorithm, we update beliefs based on our new hidden variables
def do_maximization(sequences, hidden_variables, motif_width):
    new_beliefs = list()
    for i in range(4):
        new_row = list()
        for k in range(motif_width + 1):
            nominator = count_occurences(sequences, hidden_variables, motif_width, to_char(i), k)
            denominator = 0 # pseudo count
            for b in range(4):
                denominator += count_occurences(sequences, hidden_variables, motif_width, to_char(b), k)
            new_row.append(nominator/denominator)
        new_beliefs.append(new_row)
    return new_beliefs

def print_motif_and_positions(hidden_variables, beliefs, motif_width):

    for i in range(len(hidden_variables)):
        print("motive location for sequence " + i.__str__() + ":", hidden_variables[i].index(max(hidden_variables[i])))
    maximums = [0 for _ in range(motif_width)]
    motif = ['A' for _ in range(motif_width)]
    for i in range(len(beliefs)):
        for k in range(1, len(beliefs[i])):
            if beliefs[i][k] > maximums[k - 1]:
                motif[k - 1] = to_char(i)
                maximums[k - 1] = beliefs[i][k]
    str1 = ""
    print("found motif: ", str1.join(motif))

# iteravely runs em until a certain treshhold
def run_em(sequences, motif_width):
    old_beliefs = initialize_beliefs(motif_width)
    print("initial beliefs:", old_beliefs)
    while True:
        hidden_variables = do_expectation(sequences, old_beliefs, motif_width)
        new_beliefs = do_maximization(sequences, hidden_variables, motif_width)
        # print("new variables: ", hidden_variables)
        # print("new beliefs: ", new_beliefs)
        # print("difference with old:", difference_in(old_beliefs, new_beliefs, motif_width))
        if difference_in(old_beliefs, new_beliefs, motif_width) > 0.0001:
            old_beliefs = new_beliefs
        else:
            print_motif_and_positions(hidden_variables, new_beliefs, motif_width)
            return hidden_variables, new_beliefs

# print(do_expectation(["GCTGTAG", "CTGCTAG"], p, 3))
test_sequences = ["TTTGGGGCTGTG", "CCCTTTGAAAAA", "AATTTAACTAAG"]
run_em(test_sequences,3)