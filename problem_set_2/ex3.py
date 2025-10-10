from joblib import Parallel, delayed
from Bio import SeqIO
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# -----3a. Parsing and k-mer generation-----
with open("problem_set_2/data/human_g1k_v37.fasta", "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        sequence_chr1 = record.seq
        break

sequence_chr1 = str(sequence_chr1)
len_seq = len(sequence_chr1)
kmer_set = set()
k = 15

# Add all kmers to a set
cur_kmer = sequence_chr1[:k]
kmer_set.add(cur_kmer)
for i in tqdm(range(k, len_seq), desc='Adding kmers...'):
    cur_kmer = cur_kmer[1:] + sequence_chr1[i]
    kmer_set.add(cur_kmer)

# Exclude any 15-mer that contains more than two Ns
kmer2discard = []
for kmer in tqdm(kmer_set, desc='Cleaning kmers...'):
    if kmer.count('N') > 2:
        kmer2discard.append(kmer)
print(f'{len(kmer2discard)} kmers discarded.')
for kmer in kmer2discard:
    kmer_set.discard(kmer)

print(f'Length of Chromosome 1: {len_seq}')
print(f'Number of valid 15-mers: {len(kmer_set)}')

# -----3b. Implementing a hash family-----
M = 2**61 - 1
c = {'A':1, 'C':2, 'G':3, 'T':4, 'N':5}

# Remove leading and trailing Ns, and keep up to 2 Ns at each end
head_N_num = tail_N_num = 0
for i in range(len(sequence_chr1)):
    if sequence_chr1[i] == 'N':
        head_N_num += 1
    else:
        break
for i in range(len(sequence_chr1)-1, 0, -1):
    if sequence_chr1[i] == 'N':
        tail_N_num += 1
    else:
        break
sequence_chr1 = sequence_chr1[head_N_num-2:len(sequence_chr1)-tail_N_num+2]
print(f'Head Ns: {head_N_num}, Tail Ns: {tail_N_num}')

encoded_sequence = [c.get(char, 5) for char in sequence_chr1]

def hash_func(a, encoded_sequence, k):
    hash = 0
    num_N = encoded_sequence[:k].count(5)
    min_hash = M
    for i in range(k):
        hash = (hash * a + encoded_sequence[i]) % M
    if num_N <= 2:
        min_hash = hash
    a_pow = pow(a, k-1, M)
    for i in range(len(encoded_sequence)-k):
        next_digit = encoded_sequence[i+k]
        first_digit = encoded_sequence[i]
        
        term_to_remove = (a_pow * first_digit) % M
        hash = ((hash - term_to_remove + M) * a + next_digit) % M
        if next_digit == 5:
            num_N += 1
        if first_digit == 5:
            num_N -= 1
        if num_N > 2:
            continue
        
        min_hash = min(min_hash, hash)
    return min_hash / M

num_a_list = [1, 2, 5, 10, 100]
min_hash_list = []
a_min = int((M/6)**(1/14))
a_min += (a_min+1)%2

for num_a in tqdm(num_a_list, desc='Estimating distinct kmers...'):
    a_list = [random.randrange(a_min, M, 2) for _ in range(num_a)]    
    sum_hash = 0
    cur_min_hash = []
    # for a in a_list:
    #     cur_min_hash.append(hash_func(a, encoded_sequence, k))
    
    # Use parallel processing to speed up
    cur_min_hash = Parallel(n_jobs=-1)(
    delayed(hash_func)(a, encoded_sequence, k) for a in a_list)

    min_hash_list.append(cur_min_hash)

# -----Visualization-----
mean_est_nums = []
plt.figure(figsize=(12, 8))

for i, num_a in tqdm(enumerate(num_a_list), desc='Plotting...'):
    # Scatter plot for all hash values
    a_vals = [num_a] * len(min_hash_list[i])
    est_vals = [1/h - 1 for h in min_hash_list[i]]
    plt.scatter(a_vals, est_vals, alpha=0.2, label=f'{num_a} hashes')
    
    # Calculate and store mean
    if est_vals:
        mean_est = 1/(sum(min_hash_list[i]) / len(est_vals)) - 1
        mean_est_nums.append(mean_est)

plt.scatter(num_a_list, mean_est_nums, color='red', alpha=0.8, s=80, zorder=5, label='Mean Estimate (calculated from mean minima hash)')
plt.plot(num_a_list, mean_est_nums, color='red', alpha=0.8, linestyle='--', zorder=5)
plt.axhline(y=len(kmer_set), color='green', linestyle='-', label=f'Actual Number of Distinct 15-mers={len(kmer_set)}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Hashes')
plt.ylabel('Estimated Number of Distinct k-mers')
plt.title('Estimated vs. Actual Number of Distinct k-mers')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
plt.savefig('problem_set_2/figures/estimated_values_vs_num_hashes.png')
# plt.close()