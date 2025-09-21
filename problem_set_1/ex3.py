import random
import time
import matplotlib.pyplot as plt

def alg1(data):
  data = list(data)
  changes = True
  while changes:
    changes = False
    for i in range(len(data) - 1):
      if data[i + 1] < data[i]:
        data[i], data[i + 1] = data[i + 1], data[i]
        changes = True
  return data

def alg2(data):
  if len(data) <= 1:
    return data
  else:
    split = len(data) // 2
    left = iter(alg2(data[:split]))
    right = iter(alg2(data[split:]))
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(left)
    right_top = next(right)
    while True:
      if left_top < right_top:
        result.append(left_top)
        try:
          left_top = next(left)
        except StopIteration:
          # nothing remains on the left; add the right + return
          return result + [right_top] + list(right)
      else:
        result.append(right_top)
        try:
          right_top = next(right)
        except StopIteration:
          # nothing remains on the right; add the left + return
          return result + [left_top] + list(left)
        
def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def data2(n):
    return list(range(n))

def data3(n):
    return list(range(n, 0, -1))

# -----3a. Hypothesize the Operation-----
n = 5
print(data1(n),alg1(data1(n)))
print(data2(n),alg1(data2(n)))
print(data3(n),alg1(data3(n)))
print(random.sample(range(n), n),alg1(random.sample(range(n), n)))
print(random.sample(range(n), n),alg2(random.sample(range(n), n)))

# -----3c. Performance Measurement and Analysis-----
def time_cal(data, alg):
    shortest = float('inf')
    for _ in range(5):
        start = time.perf_counter()
        alg(data)
        end = time.perf_counter()
        if end - start < shortest:
            shortest = end - start
    return shortest

nn = 8
n = [3**i for i in range(1, nn + 1)]
alg1_times = [time_cal(data1(i), alg1) for i in n]
alg2_times = [time_cal(data1(i), alg2) for i in n]

# log-log plot
plt.figure()
plt.loglog(n, alg1_times, label='alg1')
plt.loglog(n, alg2_times, label='alg2')
plt.xlabel('Input size (n)')
plt.ylabel('Time (seconds)')
plt.title('Performance of alg1 and alg2 on data1')
plt.legend()
# plt.show()
plt.savefig('problem_set_1/figures/alg_performance.png')

# data2
alg1_times = [time_cal(data2(i), alg1) for i in n]
alg2_times = [time_cal(data2(i), alg2) for i in n]

# log-log plot for data2
plt.figure()
plt.loglog(n, alg1_times, label='alg1')
plt.loglog(n, alg2_times, label='alg2')
plt.xlabel('Input size (n)')
plt.ylabel('Time (seconds)')
plt.title('Performance of alg1 and alg2 on data2')
plt.legend()
# plt.show()
plt.savefig('problem_set_1/figures/alg_performance_data2.png')

# data3
alg1_times = [time_cal(data3(i), alg1) for i in n]
alg2_times = [time_cal(data3(i), alg2) for i in n]  
# log-log plot for data3
plt.figure()
plt.loglog(n, alg1_times, label='alg1')
plt.loglog(n, alg2_times, label='alg2')
plt.xlabel('Input size (n)')
plt.ylabel('Time (seconds)')    
plt.title('Performance of alg1 and alg2 on data3')
plt.legend()
# plt.show()
plt.savefig('problem_set_1/figures/alg_performance_data3.png')
