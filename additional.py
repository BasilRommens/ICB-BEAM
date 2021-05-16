import csv

import numpy

from analyse import get_fasta_data_list, clean_up_strings, BASES, \
    count_occurrence


def score_motif(instances, motif):
    score = 0
    for i in range(len(instances)):
        temp_score = 0
        for j in range(len(instances[i]) - len(motif)):
            new_score = sum(
                [instances[i][j + k] == motif[k] for k in range(len(motif))])
            temp_score = max(temp_score, new_score)
        score += temp_score
    return score, count_occurrence(instances, motif)


def filter_motifs(collection):
    motifs = list()
    for element in collection:
        if element[0] not in BASES: continue
        motifs.append(element)
    return motifs


def custom_max(elements):
    max_list = set()
    max_num = 0
    for element in elements:
        if element[0][0] > max_num:
            max_num = element[0][0]
            max_list.clear()
            max_list.add(element)
        if element[0][0] == max_num:
            max_list.add(element)
    return max_list


def print_max(instances, motifs):
    scores = [(score_motif(instances, motif), motif) for motif in motifs]
    max_scores = custom_max(scores)
    length = len(motifs[0])
    num_instances = len(instances)
    for max_score in max_scores:
        print(
            f"{max_score[1]} ({max_score[0][0] / (num_instances * length) * 100}%, {max_score[0][1]})")


def print_avg(instances, motifs):
    length = len(motifs[0])
    print(length)
    num_instances = len(instances)
    scores = [score_motif(instances, motif)[0] / (num_instances * length) * 100
              for motif in motifs]
    avg_score = numpy.median(scores)
    print(f"{avg_score}%")


def print_std(instances, motifs):
    length = len(motifs[0])
    print(length)
    num_instances = len(instances)
    scores = [score_motif(instances, motif)[0] / (num_instances * length) * 100
              for motif in motifs]
    std_score = numpy.std(scores)
    print(f"{std_score}%")


data_file_name = "testdata_16S_RNA.FASTA"
temp_instances = get_fasta_data_list(data_file_name)
instances = clean_up_strings(temp_instances)
file_names = ["G.csv", "BOG.csv", "EM.csv", "BOEM.csv"]
interval = [10, 15]
length = 20
num_instances = len(instances)

for file_name in file_names:
    print(file_name)
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        motifs = list()
        max_motifs = list()
        for row in csv_reader:
            line_count += 1
            if line_count < interval[0] + 1:
                continue
            if line_count == interval[1] + 1:
                break
            new_motifs = filter_motifs(row[3].split('\''))
            scores = [(score_motif(instances, motif), motif) for motif in
                      new_motifs]
            max_motif = list(custom_max(scores))[0][1]
            motifs.extend(new_motifs)
            max_motifs.append(max_motif)

    print_max(instances, motifs)
    print_avg(instances, max_motifs)
    print_std(instances, max_motifs)
