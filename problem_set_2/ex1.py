from bitarray import bitarray
from hashlib import sha3_256, sha256, blake2b
import json
import pandas as pd
import matplotlib.pyplot as plt

def my_hash(s, size):
    return int(sha256(s.lower().encode()).hexdigest(), 16) % size

def my_hash2(s, size):
    return int(blake2b(s.lower().encode()).hexdigest(), 16) % size

def my_hash3(s, size):
    return int(sha3_256(s.lower().encode()).hexdigest(), 16) % size


class BloomFilter:
    def __init__(self, size, hash_functions):
        self.size = size
        self.hash_functions = hash_functions
        # Initialize all bits to 0 (False).
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        for h_func in self.hash_functions:
            index = h_func(item, self.size)
            self.bit_array[index] = 1

    def __contains__(self, item):
        # If all bits were 1, the item is probably in the set.
        for h_func in self.hash_functions:
            index = h_func(item, self.size)
            if not self.bit_array[index]:
                return False
        return True
    

# -----1a. Implementing and Populate a Bloom Filter-----
FILTER_SIZE = int(1e7)
HASH_FUNCTIONS = [my_hash, my_hash2, my_hash3]
bloom_filter = BloomFilter(FILTER_SIZE, HASH_FUNCTIONS)
word_set = set()
word_count = 0
with open('data/words.txt') as f:
    for line in f:
        word = line.strip()
        if word:
            bloom_filter.add(word)
            word_set.add(word)
            word_count += 1
print(f"Added {word_count} words to the Bloom filter.")

# -----1b. Spell Check and Correction-----
def spelling_correction(word, bloom_filter):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    suggestions = []

    # Check for single character edits
    for i in range(len(word)):
        for c in alphabet:
            candidate = word[:i] + c + word[i+1:]
            if candidate in bloom_filter:
                suggestions.append(candidate)

    return suggestions
# test
# print(spelling_correction('floeer', bloom_filter))

def evaluation(bloom_filter, typos):
    good_suggestions = 0
    false_positives = 0
    total_typo = 0

    for [typed_word, correct_word] in typos:
        # Spell check
        if typed_word == correct_word:
            continue
        # Spell correction
        total_typo += 1
        suggestions = spelling_correction(typed_word, bloom_filter)
        if len(suggestions)<=3 and correct_word in suggestions:
            good_suggestions += 1
        if typed_word in bloom_filter:
            false_positives += 1
            
    return good_suggestions, total_typo, false_positives

with open('data/typos.json', 'r') as file:
    typos = json.load(file)
good_suggestions, total_typo, false_positives = evaluation(bloom_filter, typos)

print(f"Ratio of 'good' suggestions: {good_suggestions}/{total_typo}={good_suggestions/total_typo:.4f}")

# -----1c. Analysis and Reflection-----
HASH_FUNCTIONS_COMB = [[my_hash], [my_hash, my_hash2], 
                       [my_hash, my_hash2, my_hash3]]
FILTER_SIZES = [int(10**i) for i in range(1, 10)]
results = []

for HASH_FUNCTIONS in HASH_FUNCTIONS_COMB:
    for FILTER_SIZE in FILTER_SIZES:
        bloom_filter = BloomFilter(FILTER_SIZE, HASH_FUNCTIONS)
        for word in word_set:
            bloom_filter.add(word)
        good_suggestions, total_typo, false_positives = evaluation(bloom_filter, typos)
        results.append({
            'num_hashes': len(HASH_FUNCTIONS),
            'filter_size': FILTER_SIZE,
            'good_suggestion_rate': good_suggestions/total_typo,
            'fp_rate': false_positives/total_typo
        })

df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(12, 8))

for num_hashes in sorted(df['num_hashes'].unique()):
    subset = df[df['num_hashes'] == num_hashes]
    
    # Plot FP Rate 
    ax.plot(subset['filter_size'], subset['fp_rate'], marker='o', linestyle='-', 
             label=f'Misidentified %, {num_hashes} Hashes')
             
    # Plot Good Suggestion Rate 
    ax.plot(subset['filter_size'], subset['good_suggestion_rate'], marker='x', linestyle='--', 
             label=f'Good Suggestion %, {num_hashes} Hashes')

ax.set_xlabel('Bits in Bloom Filter')
ax.set_xscale('log')
ax.set_ylim(0, 1.05) 
ax.legend()
plt.title('Bloom Filter Performance vs. Size and Hash Functions')
fig.tight_layout()
# plt.show()
plt.savefig('figures/bloom_filter_performance.png')