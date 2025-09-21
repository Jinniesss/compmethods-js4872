import numpy as np
import time
import matplotlib.pyplot as plt

class Tree:
    def __init__(self):
        self._value = None
        self._data = None
        self.left = None
        self.right = None
    def add(self,value, data):
        if self._value is None:
            self._value = value
            self._data = data
        elif value < self._value:
            if self.left is None:
                self.left = Tree()
            self.left.add(value, data)
        else:
            if self.right is None:
                self.right = Tree()
            self.right.add(value, data)

    def __contains__(self, patient_id):
        if self._value == patient_id:
            return True
        elif self.left and patient_id < self._value:
            return patient_id in self.left
        elif self.right and patient_id > self._value:
            return patient_id in self.right
        else:
            return False
    def has_data(self, data):
        if self._data == data:
            return True
        elif self.left and self.left.has_data(data):
            return True
        elif self.right and self.right.has_data(data):
            return True
        else:
            return False
        

# -----4a. Implement the add Method-----
# my_tree = Tree()
# for patient_id, initials in [(24601, "JV"), (42, "DA"), (7, "JB"), (143, "FR"), (8675309, "JNY")]:
#     my_tree.add(patient_id, initials)
# # print(my_tree.left._value)

# # -----4b. Implement a __contains__ Method-----
# print(24601 in my_tree)
# print(1492 in my_tree)

# # -----4c. Implement and Test a has_data Method-----
# print(my_tree.has_data("JV"))
# print(my_tree.has_data(24601))

# -----4d. Performance Analysis of __contains__ and has_data-----
n_values = np.logspace(1, 4, num=7, dtype=int)
contains_times = []
has_data_times = []
for n in n_values:
    my_tree = Tree()
    for patient_id in np.random.permutation(n):
        my_tree.add(patient_id, f"Patient {patient_id}")
    
    # Time __contains__ method
    shortest_contains_time = float('inf')
    for _ in range(5):
        start_time = time.perf_counter()
        for patient_id in range(n):
            _ = patient_id in my_tree
        cur_contain_time = time.perf_counter() - start_time
        if cur_contain_time < shortest_contains_time:
            shortest_contains_time = cur_contain_time
    contains_times.append(shortest_contains_time/n)
    
    # Time has_data method
    shortest_hasdata_time = float('inf')
    for _ in range(5):
        start_time = time.perf_counter()
        for patient_id in range(n):
            _ = my_tree.has_data(f"Patient {patient_id}")
        cur_hasdata_time = time.perf_counter() - start_time
        if cur_hasdata_time < shortest_hasdata_time:
            shortest_hasdata_time = cur_hasdata_time
    has_data_times.append(shortest_hasdata_time/n)

# Reference line for O(log n)
C = contains_times[0] / np.log(n_values[0])
log_ref_times = C * np.log(n_values)

plt.figure()
plt.loglog(n_values, contains_times, label='contains')
plt.loglog(n_values, has_data_times, label='has_data')
plt.loglog(n_values, log_ref_times, '--', label='O(log N) reference')
plt.xlabel('Number of nodes in tree')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title('Performance of __contains__ vs has_data')
# plt.show()
plt.savefig('problem_set_1/figures/tree_performance.png')

# Set-up analysis
setup_times = []
for n in n_values:
    shortest_setup_time = float('inf')
    for _ in range(5): 
        start_time = time.perf_counter()
        my_tree = Tree()
        for patient_id in np.random.permutation(n):
            my_tree.add(patient_id, f"Patient {patient_id}")
        cur_setup_time = time.perf_counter() - start_time
        if cur_setup_time < shortest_setup_time:
            shortest_setup_time = cur_setup_time
    setup_times.append(shortest_setup_time)

# Reference line for O(n)
C_setup = setup_times[0] / n_values[0]
linear_ref_times = C_setup * n_values
# Reference line for O(n^2)
quadratic_ref_times = C_setup * (n_values**2) / n_values[0]

plt.figure()
plt.loglog(n_values, setup_times, label='setup')
plt.loglog(n_values, linear_ref_times, '--', label='O(N) reference')
plt.loglog(n_values, quadratic_ref_times, ':', label='O(N^2) reference')
plt.xlabel('Number of nodes in tree')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title('Performance of Tree Setup')
# plt.show()
plt.savefig('problem_set_1/figures/tree_setup_performance.png')