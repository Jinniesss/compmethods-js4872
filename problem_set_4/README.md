# CBB 6340 - Problem Set 4

### Student Information

Name: Jinnie Sun

NetID: js4872

### Exercise Answers & Results

#### Exercise 1: Gradient Descent for Neural Network Parameter Optimization (25 points)

Is this API access point a [Clean URL](https://en.wikipedia.org/wiki/Clean_URL)? Why or why not? (**1 point**)

+ No.
+ It includes the file path, script name and query parameters.

Implement a two-dimensional version of the gradient descent algorithm to find optimal choices of a and b. (**6 points**)

+ ```python
  ite = 100
  epsilon = 1e-9
  lr = 0.4
  decay = 0.9
  init_ite = 10
  minima = []
  
  for j in range(init_ite):
      a = random.uniform(0,1)
      b = random.uniform(0,1)
      lr = 0.4
      last_err = float('inf')
      for i in range(ite):
          cur_err = err(a, b)
          print(f"Iteration {i}: a = {a}, b = {b}, error = {cur_err}")
          da = (err(a + epsilon, b) - cur_err)/epsilon
          db = (err(a, b + epsilon) - cur_err)/epsilon
          a = min(a - lr * da, 1-2*epsilon)
          b = max(b - lr * db, 2*epsilon)
  
          if abs(last_err - cur_err) < epsilon:
              break
          last_err = cur_err
          lr *= decay
  ```

Explain how you estimate the gradient given that you cannot directly compute the derivative (**3 points**), identify any numerical choices -- including but not limited to stopping criteria -- you made (**3 points**), and justify why you think they were reasonable choices (**3 points**).

+ Gradient estimation: Forward difference method is used to estimate the partial gradient for a and b separately, with `epsilon=1e-16`, which is small enough to provide a close approximation. 
+ Learning: with a learning rate of `0.4`, which allows fast convergence in this case, and a decay rate of `0.9`, which enables it to approach the minima closer. Also it's ensured that `a` and `b` are within the valid range.
+ Stopping criteria: It stops when reaching a maximum number of iterations or when the error values do not change very much. This ensures that the function will terminate and save time if it's not making improvements.

Find both locations (i.e. a, b values) querying the API as needed (**5 points**) and identify which corresponds to which (**2 points**). Briefly discuss how you would have tested for local vs global minima if you had not known how many minima there were. (**2 points**)

+ local minimum: a=0.22, b=0.69

  global minimum: a=0.71, b=0.17

+ I tried different (in this case 10) initial conditions of a and b, ran the gradient descent separately, and then compared their error values: 

  ![Figure_1](assets/Figure_1.png)

  The one with lowest error value would be considered the global minimum.


### Exercise 2: Sequence Alignment

Implement a function that takes two strings and returns an optimal local alignment (**6 points**) and score (**6 points**) using the Smith-Waterman algorithm.

+ ```python
  def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
      m = len(seq1)
      n = len(seq2)
  
      # Initialize and calculate scoring matrix
      score = [[0 for j in range(n+1)] for i in range(m+1)]
      max_score = 0
      max_pos = (0, 0)
      for i in range(1, m+1):
          for j in range(1, n+1):
              if seq1[i-1] == seq2[j-1]:
                  diag = score[i-1][j-1] + match
              else:
                  diag = score[i-1][j-1] - mismatch_penalty
              up = score[i-1][j] - gap_penalty
              left = score[i][j-1] - gap_penalty
              score[i][j] = max(0, diag, up, left)
              if score[i][j] > max_score:
                  max_score = score[i][j]
                  max_pos = (i, j)
  
      # Traceback
      aligned_seq1 = []
      aligned_seq2 = []
      i, j = max_pos
      while score[i][j] != 0: 
          if score[i][j] == score[i-1][j] - gap_penalty:
              aligned_seq1.append(seq1[i-1])
              aligned_seq2.append('-')
              i -= 1
          elif score[i][j] == score[i][j-1] - gap_penalty:
              aligned_seq1.append('-')
              aligned_seq2.append(seq2[j-1])
              j -= 1
          else:
              aligned_seq1.append(seq1[i-1])
              aligned_seq2.append(seq2[j-1])
              i -= 1
              j -= 1
  
      aligned_seq1.reverse()
      aligned_seq2.reverse()
  
      return ''.join(aligned_seq1), ''.join(aligned_seq2), max_score
  ```

Testing:

+ test 1

  ```python
  print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac'))
  ```

  result 1

  ```python
  ('agacccta-cgt-gac', 'agacc-tagcatcgac', 8)
  ```

+ test 2

  ```python
  print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2))
  ```

  result 2

  ```python
  ('gcatcga', 'gcatcga', 7)
  ```

+ test 3

  ```python
  print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', mismatch_penalty=0.3))
  ```

  result 3

  ```python
  ('cgagacccta-cgt-gac', 'ctagacc-tagcatcgac', 9.4)
  ```

+ The first two test results align with the provided examples. A higher gap penalty (test 2) results in fewer gaps in the alignment, while a lower mismatch penalty (test 3) allows for a greater number of mismatches. 



