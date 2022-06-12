# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

# %%
naive_model = GaussianNB().fit(X_train, y_train)

# %%
train_sc = naive_model.score(X_train, y_train)
test_sc = naive_model.score(X_test, y_test)

print('Train accuracy: {}'.format(train_sc))
print('Test accuracy: {}'.format(test_sc))