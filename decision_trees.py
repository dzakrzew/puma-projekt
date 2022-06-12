# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

# %%
data = np.genfromtxt('card_transdata.csv', delimiter=',', skip_header=1)

# %%
X = data[:, :-1]
y = data[:, -1]

# %%
# Podgląd danych
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

classifier = tree.DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# %%
path = classifier.cost_complexity_pruning_path(X_train, y_train)
print(path)

# %%
alphas = path.ccp_alphas
print(classifier.score(X_test, y_test))
print(classifier.get_depth())

# %%
classifiers = []
depths = []

for alpha in alphas:
    classifier = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    classifier.fit(X_train, y_train)
    classifiers.append(classifier)
    depths.append(classifier.get_depth())

# %%
train_scores = [clf.score(X_train, y_train) for clf in classifiers]
test_scores = [clf.score(X_test, y_test) for clf in classifiers]
print(test_scores)

# %%
classifiers = classifiers[:-1]
depths = depths[:-1]

# %%
fig, ax = plt.subplots()
ax.set_xlabel('Depths')
ax.set_ylabel('Accuracy')
ax.plot(depths, train_scores[:-1], marker='o', label='train')
ax.plot(depths, test_scores[:-1], marker='o', label='test')
ax.legend()
plt.show()

# %%
