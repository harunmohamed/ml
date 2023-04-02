# scikit-learn syntax
'''

from sklearn.module import Model
model  = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)


'''

# k nearest neighbors
'''

from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_evening_charge"]].values
y = churn_df["churn"].values
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
predictions  = knn.predict(X_new)
print('Predictions: {}'.format(predictions)) // Predictions: [1 0 0]

'''

# measuring model performance
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

"""

# model complexity and over / underfitting
"""

train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

##  plotting model complexity curve
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()


"""
