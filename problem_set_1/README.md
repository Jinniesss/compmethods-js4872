# CBB 6340 - Problem Set 1

### Student Information

Name: Jinnie Sun

NetID: js4872

### Instructions for Running Scripts

This project's scripts are written in Python.

### Exercise Answers & Results

#### Exercise 1: Efficiently search patient data

##### 1a. **Plot Age **Distribution

+ Plot a histogram showing the distribution of ages:

![age_distribution](figures/age_distribution.png)

+ No patients share the same exact age, as converting the list of ages to a set does not change the total count.

  Command:

  ```python
  print(len(set(ages)), len(ages))
  ```

  Output:

  ```python
  324357 324357
  ```

+ *Extra Credit:* (2 points) Explain how the existence (or non-existence) of multiple patients with the same age affects the solution to the rest of the problem.

  【tbc】

##### 1b. Plot Gender Distribution

+ Plot the distribution of genders from the dataset.

  ![gender_distribution](figures/gender_distribution.png)

+ Identify how gender is encoded in the data and list the categories used.

  Gender is encoded by a string. Three categories are used: 'female', 'male', and 'unknown'.

##### 1c. Sort Patients by Age

Eldest patient: Monica Caponera, Age 84.99855742449432, Gender female

##### 1d. Finding the Second Oldest Patient

- Describe a method to find the second oldest patient in O(n) time. Keep in mind sorting the list is O(n log n).

  ```python
  oldest = 0
  second_oldest = 0
  for patient in patients:
  	if patient.age > oldest:
  		second_oldest = oldest
      oldest = patient.age
    elif patient.age > second_oldest:
      second_oldest = patient.age
  ```

- Discuss scenarios where it is advantageous to sort the data versus using the O(n) solution.

  Sorting the data is advantageous when we need to make multiple rank-based queries later. While if we only need one or a few specific queries, such as the largest and the second largest items, the O(n) solution is more efficient. 

##### 1e. Binary Search for Specific Age

```python
# ----- 1e. Binary Search for Specific Age-----
def binary_search(patients, target_age):
    low = 0
    high = len(patients) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age == target_age:
            return patients[mid]
        elif mid_age < target_age:
            low = mid + 1
        else:
            high = mid - 1

    return None

patient_41d5 = binary_search(patients_sorted_by_age, 41.5)
print(f"Patient aged 41.5: {patient_41d5.get('name')}, Age {patient_41d5.get('age')}, Gender {patient_41d5.get('gender')}")
```

Output:

```python
Patient aged 41.5: John Braswell, Age 41.5, Gender male
```

##### 1f. Count Patients Above a Certain Age

With only small modifications on `binary_search`.

```python
# -----1f. Count Patients Above a Certain Age-----
def count_above_age(patients, target_age):
    count = 0
    low = 0
    high = len(patients) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age == target_age:
            count += high - mid
            return count
        elif mid_age < target_age:
            low = mid + 1
        else:
            count += high - mid + 1
            high = mid - 1
        
    return count

count_above_41d5 = count_above_age(patients_sorted_by_age, 41.5)
print(f"Number of patients older than 41.5: {count_above_41d5}")
# validate with a linear scan
# print(len([age for age in ages if age > 41.5]))
```

Output:

```python
Number of patients older than 41.5: 150470
```

##### 1g. Function for Age Range Query

```python
# -----1g. Function for Age Range Query-----
def count_in_age_range(patients, low_age, high_age):
    count = 0
    low = 0
    high = len(patients) - 1

    # Find the first patient >= low_age
    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age < low_age:
            low = mid + 1
        else:
            high = mid - 1

    start_index = low

    # Find the last patient < high_age
    low = 0
    high = len(patients) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_age = float(patients[mid].get('age'))

        if mid_age < high_age:
            low = mid + 1
        else:
            high = mid - 1

    end_index = high

    if start_index <= end_index:
        count = end_index - start_index + 1

    return count

# validation
print(count_in_age_range(patients_sorted_by_age, 0, 100), len(ages))
print(count_in_age_range(patients_sorted_by_age, 30, 41.5), len([age for age in ages if 30 <= age < 41.5]))
```

Output:

```python
324357 324357
49752 49752
```

##### 1h. Function for Age and Gender Range Query

```python
# -----1h. Function for Age and Gender Range Query-----
def count_male_in_age_range(patients, low_age, high_age):
    # data setup
    patients_male = [p for p in patients if p.get('gender') == 'male']
    
    count = count_in_age_range(patients, low_age, high_age)
    count_male = count_in_age_range(patients_male, low_age, high_age)

    return count, count_male

# validation
print(count_male_in_age_range(patients_sorted_by_age, 0, 100), (len(ages), len([p for p in patients if p.get('gender') == 'male' and 0 <= float(p.get('age')) < 100])))
print(count_male_in_age_range(patients_sorted_by_age, 30, 41.5), (len([age for age in ages if 30 <= age < 41.5]), len([p for p in patients if p.get('gender') == 'male' and 30 <= float(p.get('age')) < 41.5])))
```

Output:

```python
(324357, 158992) (324357, 158992)
(49752, 24869) (49752, 24869)
```