import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

data = pd.read_excel('Rice_Cammeo_Osmancik.xlsx')
train_idx, test_idx = train_test_split(data.index, test_size=0.1, stratify=data['Class'])
data_train = data.loc[train_idx].copy()
data_test = data.loc[test_idx].copy()

# Normalize
my_cols = [col for col in data.columns if col != 'Class' and col != 'train']
for column in my_cols:
    mean = data_train[column].mean()
    std = data_train[column].std()
    data_train.loc[:, column] = (data_train[column] - mean) / std
    data_test.loc[:, column] = (data_test[column] - mean) / std

# PCA
pca = decomposition.PCA(n_components=2)
train_data_reduced = pca.fit_transform(data_train[my_cols])
pc0 = train_data_reduced[:, 0]
pc1 = train_data_reduced[:, 1]
test_data_reduced = pca.transform(data_test[my_cols])

# Plotting
# classes = data['Class'].unique()
# colors = ['r', 'g', 'b']
# for cls, color in zip(classes, colors):
#     indices = data['Class'] == cls
#     plt.scatter(pc0[indices], pc1[indices], c=color, label=cls, alpha=0.5, s=0.5)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# # plt.show()
# plt.savefig('pca.png')

class Point:
    def __init__(self, x, y, class_label):
        self.x = x
        self.y = y
        self.class_label = class_label

class QuadTree:    
    CAPACITY = 10       # maximum number of points in a leaf node before splitting
    
    def __init__(self, xlo, ylo, xhi, yhi, parent=None):
        self.xlo, self.ylo = xlo, ylo
        self.xhi, self.yhi = xhi, yhi
        self.parent = parent
        self.points = [] 
        self.children = None
        
    def contains(self, x, y):
        return (self.xlo <= x < self.xhi) and (self.ylo <= y < self.yhi)
    
    def insert(self, point):
        """Inserts a Point object into the tree, recursively splitting the node."""
        if not self.contains(point.x, point.y):
            return False 

        if self.children is None:
            # leaf node
            if len(self.points) < self.CAPACITY:
                self.points.append(point)
                return True
            else:
                # split the node
                self._subdivide()
                all_points = self.points + [point]
                self.points = [] 
                
                for p in all_points:
                    for child in self.children.values():
                        if child.insert(p):
                            break
                return True
        else:
            # Not a leaf node
            for child in self.children.values():
                if child.contains(point.x, point.y):
                    return child.insert(point)

    def _subdivide(self):
        # split the node into 4 children
        mid_x = (self.xlo + self.xhi) / 2
        mid_y = (self.ylo + self.yhi) / 2
        
        self.children = {
            'NW': QuadTree(self.xlo, mid_y, mid_x, self.yhi, self),
            'NE': QuadTree(mid_x, mid_y, self.xhi, self.yhi, self),
            'SW': QuadTree(self.xlo, self.ylo, mid_x, mid_y, self),
            'SE': QuadTree(mid_x, self.ylo, self.xhi, mid_y, self),
        }
        
    def _within_distance(self, x, y, d):
        # the closest point on the box to (x, y)
        closest_x = max(self.xlo, min(x, self.xhi))
        closest_y = max(self.ylo, min(y, self.yhi))
    
        distance_sq = (x - closest_x)**2 + (y - closest_y)**2
        return distance_sq <= d**2
        
    def leaves_within_distance(self, x, y, d):
        if not self._within_distance(x, y, d):
            return []
        
        if self.children is None:
            # leaf node and within distance
            return [self]
        else:
            # Not a leaf node, check children recursively
            results = []
            for child in self.children.values():
                results.extend(child.leaves_within_distance(x, y, d))
            return results

    def small_containing_quadtree(self, x, y):
        # Finds the smallest leaf node that contains (x, y).
        if self.children is None:
            return self
        
        for child in self.children.values():
            if child.contains(x, y):
                return child.small_containing_quadtree(x, y)
        return self
    
X_train = train_data_reduced
y_train = data_train["Class"].values

X_test = test_data_reduced
y_test = data_test["Class"].values

# QuadTree Construction
xlo, xhi = np.min(X_train[:, 0]), np.max(X_train[:, 0])
ylo, yhi = np.min(X_train[:, 1]), np.max(X_train[:, 1])

buffer = 1e-6 
root_quadtree = QuadTree(xlo, ylo, xhi + buffer, yhi + buffer)

for i in range(len(y_train)):
    point = Point(X_train[i, 0], X_train[i, 1], y_train[i])
    root_quadtree.insert(point)
    
print(f"QuadTree built with {len(X_train)} training points.")

def quadtree_k_nearest(root, x, y, k):
    # Return k nearest Points to (x,y)
    
    # expand radius until we have at least k candidate points
    d = 1e-6
    max_d = np.hypot(root.xhi - root.xlo, root.yhi - root.ylo) * 2

    candidates = []
    while True:
        leaves = root.leaves_within_distance(x, y, d)
        candidates = []
        for leaf in leaves:
            candidates.extend(leaf.points)

        if len(candidates) >= k or d >= max_d:
            break
        d = max(d * 2, 1e-6)

    # if still not enough, collect all points from the tree
    if len(candidates) < k:
        stack = [root]
        candidates = []
        while stack:
            node = stack.pop()
            if node.children is None:
                candidates.extend(node.points)
            else:
                stack.extend(node.children.values())

    if len(candidates) == 0:
        return []

    distances = np.array([np.hypot(p.x - x, p.y - y) for p in candidates])
    kk = min(k, len(distances))
    idx = np.argpartition(distances, kk - 1)[:kk]
    neighbors = [candidates[i] for i in idx]
    neighbors = sorted(neighbors, key=lambda p: np.hypot(p.x - x, p.y - y))
    return neighbors

def knn_predict(root, x, y, k):
    neighbors = quadtree_k_nearest(root, x, y, k)
    if len(neighbors) == 0:
        return None
    counts = Counter([p.class_label for p in neighbors])
    most_common = counts.most_common()
    # if tie for top count, break tie by picking the class of the closest neighbor among tied classes
    top_count = most_common[0][1]
    tied = [cls for cls, cnt in most_common if cnt == top_count]
    if len(tied) == 1:
        return most_common[0][0]
    # tie-breaker: nearest neighbor's class among tied classes
    for p in neighbors:
        if p.class_label in tied:
            return p.class_label

# Evaluate on test set for k=1 and k=5
classes = data['Class'].unique()
def evaluate_k(k):
    preds = []
    for i in range(len(X_test)):
        x, y = X_test[i, 0], X_test[i, 1]
        pred = knn_predict(root_quadtree, x, y, k)
        preds.append(pred)
    preds = np.array(preds)
    cm = confusion_matrix(y_test, preds, labels=classes)
    acc = np.mean(preds == y_test)
    return cm, acc, preds

for k in [1, 5]:
    cm, acc, preds = evaluate_k(k)
    print(f"\nConfusion matrix for k={k} (rows=true, cols=predicted) :")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    print(f"Accuracy for k={k}: {acc:.3f}")