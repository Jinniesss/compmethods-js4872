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