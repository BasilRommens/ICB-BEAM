# Compares the performance of gibbs with exmin
import time
import resource

from gibbs import gibbs_sample


def get_performance(func, *args, **kwargs):
    time_start = time.perf_counter()

    func(*args, **kwargs)

    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

    return memMb, time_elapsed


if __name__ == '__main__':
    print("Performance statistics\t\t Time (s)\t\t\t Memory (Mb)")

    instances = ["CAAAACCCTCAAATACATTTTAGAAACACAATTTCAGGATATTAAAAGTTAAATTCATCTAGTTATACAA",
                 "TCTTTTCTGAATCTGAATAAATACTTTTATTCTGTAGATGGTGGCTGTAGGAATCTGTCACACAGCATGA",
                 "CCACGTGGTTAGTGGCAACCTGGTGACCCCCCTTCCTGTGATTTTTACAAATAGAGCAGCCGGCATCGTT",
                 "GGAGAGTGTTTTTAAGAAGATGACTACAGTCAAACCAGGTACAGGATTCACACTCAGGGAACACGTGTGG",
                 "TCACCATCAAACCTGAATCAAGGCAATGAGCAGGTATACATAGCCTGGATAAGGAAACCAAGGCAATGAG"]

    gibbs_time, gibbs_memory = get_performance(gibbs_sample, instances, 8)
    print(f"Our gibbs implementation:\t {gibbs_time}\t {gibbs_memory}")
