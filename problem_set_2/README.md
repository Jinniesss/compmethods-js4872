# CBB 6340 - Problem Set 2

### Student Information

Name: Jinnie Sun

NetID: js4872

### Instructions for Running Scripts

This project's scripts are written in Python.

### Exercise Answers & Results

#### Exercise 1: Spelling Correction Using a Bloom Filter

##### 1a. Implementing and Populate a Bloom Filter

+ Create a Bloom Filter

  ```python
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
  ```

+ Insert Words into the Bloom Filter

  ```python
  FILTER_SIZE = int(1e7)
  HASH_FUNCTIONS = [my_hash, my_hash2, my_hash3]
  bloom_filter = BloomFilter(FILTER_SIZE, HASH_FUNCTIONS)
  word_set = set()
  word_count = 0
  with open('words.txt') as f:
      for line in f:
          word = line.strip()
          if word:
              bloom_filter.add(word)
              word_set.add(word)
              word_count += 1
  print(f"Added {word_count} words to the Bloom filter.")
  ```

  Output

  ```python
  Added 466550 words to the Bloom filter.
  ```

##### 1b. Spell Check and Correction

+ Implement a Spelling Correction Function

  ```python
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
  ```

  Validated with self-check examples.

+ Evaluate Performance

  ```python
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
          suggestions = spelling_correction(typed_word, 
                                            bloom_filter)
          if len(suggestions)<=3 and 
          		correct_word in suggestions:
              good_suggestions += 1
          if typed_word in bloom_filter:
              false_positives += 1
              
      return good_suggestions, total_typo, false_positives
    
  with open('typos.json', 'r') as file:
      typos = json.load(file)
  good_suggestions, total_typo, false_positives = evaluation(bloom_filter, typos)
  
  print(f"Ratio of 'good' suggestions: {good_suggestions}/{total_typo}={good_suggestions/total_typo:.4f}")
  ```

  Output:

  ```python
  Ratio of 'good' suggestions: 22900/25000=0.9160
  ```

##### 1c. Analysis and Reflection

+ Plot the Effect of Filter Size and Number of Hash Functions

  ![bloom_filter_performance](figures/bloom_filter_performance.png)

+ Approximately how many bits are necessary to achieve 85% good suggestions with each combination of 1, 2, or 3 hash functions?

  3 hash function: 5e6

  2 hash functions: 3e7

  1 hash functions: 1e8

#### Exercise 2: Accelerating data processing with parallel programming

##### 2a. Modify alg2 for Keyed Sorting

I use a list of integers `address=list(range(len(data)))` to track the sorting process. The ith number in `address` represents the index of the item in the original list of data that corresponds to the ith item in `keys`. `address` always has the same length as `keys`, and transposes exactly as `keys`. 

After (`alg2_dict`) taking the argument of the data dictionary and the specified key to sort by, only the two lists of numbers (`keys` and `address`) are involved in the actual sorting algorithm (`alg2`), instead of the whole data. 

```python
import random
random.seed(0)

def alg2(keys, address):
    if len(keys) <= 1:
        return keys, address
    else:
        split = len(keys) // 2
        left, left_address = alg2(keys[:split],address[:split])
        right, right_address = alg2(keys[split:],address[split:])
        left = iter(left)
        right = iter(right)
        left_address = iter(left_address)
        right_address = iter(right_address)
        result = []
        result_address = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        left_addr_top = next(left_address)
        right_addr_top = next(right_address)
        while True:
            if left_top < right_top:
                result.append(left_top)
                result_address.append(left_addr_top)
                try:
                    left_top = next(left)
                    left_addr_top = next(left_address)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right), 
                  result_address + [right_addr_top] + list(right_address)
            else:
                result.append(right_top)
                result_address.append(right_addr_top)
                try:
                    right_top = next(right)
                    right_addr_top = next(right_address)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left), 
                  result_address + [left_addr_top] + list(left_address)
                
def alg2_dict(data, key):
    print(f'Original {key}s: {[d[key] for d in data]}')
    keys = [d[key] for d in data]
    addr = list(range(len(data)))
    keys_sorted, addr_sorted = alg2(keys, addr)
    print(f'Sorted {key}s: {keys_sorted}')
    return [data[i] for i in addr_sorted]
```

Testing examples:

+ `patient_id`, `patient_data,` and `age` are generated for each item.

  Randomly generated `patient_id` and `age` have been used as the specified key respectively. 

  ```python
  n = 5
  data = [{'patient_id': i, 'patient_data': chr(97 + i), 'age': random.randint(1, 99)} for i in random.sample(range(n), k=n)]
  
  print("Unsorted data:")
  print(data)
  print("\nSort by patient_id:")
  print(alg2_dict(data, 'patient_id'))
  print("\nSort by age:")
  print(alg2_dict(data, 'age'))
  ```

  Output

  ```python
  Unsorted data:
  [{'patient_id': 3, 'patient_data': 'd', 'age': 52}, {'patient_id': 4, 'patient_data': 'e', 'age': 39}, {'patient_id': 0, 'patient_data': 'a', 'age': 62}, {'patient_id': 1, 'patient_data': 'b', 'age': 46}, {'patient_id': 2, 'patient_data': 'c', 'age': 75}]
  
  Sort by patient_id:
  Original patient_ids: [3, 4, 0, 1, 2]
  Sorted patient_ids: [0, 1, 2, 3, 4]
  [{'patient_id': 0, 'patient_data': 'a', 'age': 62}, {'patient_id': 1, 'patient_data': 'b', 'age': 46}, {'patient_id': 2, 'patient_data': 'c', 'age': 75}, {'patient_id': 3, 'patient_data': 'd', 'age': 52}, {'patient_id': 4, 'patient_data': 'e', 'age': 39}]
  
  Sort by age:
  Original ages: [52, 39, 62, 46, 75]
  Sorted ages: [39, 46, 52, 62, 75]
  [{'patient_id': 4, 'patient_data': 'e', 'age': 39}, {'patient_id': 1, 'patient_data': 'b', 'age': 46}, {'patient_id': 3, 'patient_data': 'd', 'age': 52}, {'patient_id': 0, 'patient_data': 'a', 'age': 62}, {'patient_id': 2, 'patient_data': 'c', 'age': 75}]
  
  ```

  The result shows that the algorithm works well with both keys. The sorting result is correct, and the data are aligned well. 

##### 2b. Parallelize the Algorithm

In the parallel method, the list of data is equally divided into `num_cores` chunks. 

```python
def merge_sorted_lists(list1, list2, key):
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i][key] < list2[j][key]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

def alg2_dict_chunked(data, key, pool, num_chunks):
    chunk_size = len(data) // num_chunks
    tasks = []
    
    for i in range(num_chunks):
        chunk = data[i*chunk_size : (i+1)*chunk_size] 
        				if i < num_chunks - 1 else data[i*chunk_size :]
        tasks.append(pool.apply_async(alg2_dict, (chunk, key)))
    sorted_chunks = [task.get() for task in tasks]

    # Merge sorted chunks
    while len(sorted_chunks) > 1:
        merged_chunks = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                merged_chunks.append(merge_sorted_lists(
                  sorted_chunks[i], sorted_chunks[i + 1], key))
            else:
                merged_chunks.append(sorted_chunks[i])
        sorted_chunks = merged_chunks

    return sorted_chunks[0]
  
if __name__ == "__main__":
  ...
  with multiprocessing.Pool(processes=num_cores) as pool:
      alg2_dict_chunked(data, 'patient_id', pool, num_cores)
  ...
```

Result:

![algorithm_performance_plot](algorithm_performance_plot.png)

The time complexity of the parallel method is relatively lower as the dataset size increases. 

When `n=1e7` and `num_cores=8` :

```python
Elapsed time (parallel): 34.259061000077054 seconds
Elapsed time (serial): 68.18649779097177 seconds
```

It is quite promising that this parallel algorithm can reach 2x speedup with more cores and larger datasets. 

*Every execution time in the result is the shortest of 5 repeated runs. 

#### Exercise 3: Estimating the Number of Distinct 15-mers in Chromosome 1 using Hash Functions