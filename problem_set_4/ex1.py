import requests
import random
import matplotlib.pyplot as plt

def err(a,b):
    return float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", headers={"User-Agent": "MyScript"}).text)

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
    minima.append((a, b, cur_err))

# plot (a,b) and use color to represent error
x = [m[0] for m in minima]
y = [m[1] for m in minima]
c = [m[2] for m in minima]

plt.scatter(x, y, c=c, cmap='viridis', alpha=0.5)
plt.colorbar(label='Error')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Error function minima')
plt.show()