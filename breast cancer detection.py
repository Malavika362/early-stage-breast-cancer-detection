import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class LVQ:
    def __init__(self, num_input_neurons, num_codebook_vectors, learning_rate):
        self.num_input_neurons = num_input_neurons
        self.num_codebook_vectors = num_codebook_vectors
        self.learning_rate = learning_rate
        self.codebook_vectors = np.random.rand(num_codebook_vectors, num_input_neurons)

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                distances = np.linalg.norm(x - self.codebook_vectors, axis=1)
                winner_index = np.argmin(distances)
                if y == 0:
                    if winner_index == 0:
                        self.codebook_vectors[winner_index] += self.learning_rate * (x - self.codebook_vectors[winner_index])
                    else:
                        self.codebook_vectors[winner_index] -= self.learning_rate * (x - self.codebook_vectors[winner_index])
                else:
                    if winner_index == 1:
                        self.codebook_vectors[winner_index] += self.learning_rate * (x - self.codebook_vectors[winner_index])
                    else:
                        self.codebook_vectors[winner_index] -= self.learning_rate * (x - self.codebook_vectors[winner_index])

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.linalg.norm(x - self.codebook_vectors, axis=1)
            prediction = 0 if np.argmin(distances) == 0 else 1
            y_pred.append(prediction)
        return np.array(y_pred)

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train LVQ model
lvq = LVQ(num_input_neurons=X_train_scaled.shape[1], num_codebook_vectors=2, learning_rate=0.01)
lvq.train(X_train_scaled, y_train, epochs=100)

# Predict on the testing set
y_pred = lvq.predict(X_test_scaled)

# Calculate accuracy and confusion matrix
accuracy = np.mean(y_pred == y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
