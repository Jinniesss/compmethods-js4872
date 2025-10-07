from Bio import SeqIO
from tqdm import tqdm

# -----3a. Parsing and k-mer generation-----
with open("problem_set_2/data/human_g1k_v37.fasta", "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        sequence_chr1 = record.seq
        break
# sequence_chr1 = '1234567890'
# sequence_chr1 = str(sequence_chr1)[5000:6500]
# len_seq = len(sequence_chr1)
# kmer_set = set()
# k = 15

# # Add all kmers to a set
# cur_kmer = sequence_chr1[:k]
# kmer_set.add(cur_kmer)
# for i in tqdm(range(k, len_seq), desc='Adding kmers...'):
#     cur_kmer = cur_kmer[1:] + sequence_chr1[i]
#     kmer_set.add(cur_kmer)
# # print(kmer_set)

# # Exclude any 15-mer that contains more than two Ns
# kmer2discard = []
# for kmer in tqdm(kmer_set, desc='Cleaning kmers...'):
#     if kmer.count('N') > 2:
#         kmer2discard.append(kmer)
# print(f'{len(kmer2discard)} kmers discarded.')
# print(kmer2discard)
# for kmer in kmer2discard:
#     kmer_set.discard(kmer)

# print(f'Length of Chromosome 1: {len_seq}')
# print(f'Number of valid 15-mers: {len(kmer_set)}')

# -----3b. Implementing a hash family-----
M = int(2e61-1)
# cur_kmer = sequence_chr1[:k]
# for i in tqdm(range(k, len_seq), desc='Adding kmers...'):
#     cur_kmer = cur_kmer[1:] + sequence_chr1[i]

def hash_func(a, sequence, k):
    c = {'A':1, 'C':2, 'G':3, 'T':4}
    cur_kmer = [c[sequence[i]] for i in range(k)]
    hash = sum([cur_kmer[i]*(a**(k-1-i)) for i in range(k)])
    print(cur_kmer,hash)
    for i in range(k, len(sequence)):
        next_c = c[sequence[i]]
        hash = ((hash - a**(k-1)*cur_kmer[0]) * a + next_c) % M
        cur_kmer.pop(0)
        cur_kmer.append(next_c)
        print(cur_kmer,hash)

hash_func(10, 'ACGTACGT', 4)