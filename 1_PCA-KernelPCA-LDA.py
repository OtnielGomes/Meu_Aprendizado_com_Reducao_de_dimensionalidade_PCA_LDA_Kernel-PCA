import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Pre-processing
with open('census_data.pkl', mode='rb') as f:
    X_census, y_census = pickle.load(f)
# Reducing the classifiers
# PCA
pca = PCA(n_components=8)
X_census_pca = pca.fit_transform(X_census)
# Kernel PCA
kernel_pca = KernelPCA(n_components=8, kernel='rbf')
X_census_kernel_pca = kernel_pca.fit_transform(X_census)
# LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_census_lda = lda.fit_transform(X_census, y_census)

# Training
# PCA
neural_network_pca = MLPClassifier(hidden_layer_sizes=(5, 5),
                                   activation='relu',
                                   solver='adam',
                                   batch_size=128,
                                   max_iter=3000,
                                   tol=1e-5,
                                   verbose=True)
neural_network_pca.fit(X_census_pca, y_census)

# Prediction
prediction_pca = neural_network_pca.predict(X_census_pca)
prediction_pca_accuracy = accuracy_score(y_census, prediction_pca)
# KernelPCA
neural_network_kernel_pca = MLPClassifier(hidden_layer_sizes=(5, 5),
                                          activation='relu',
                                          solver='adam',
                                          batch_size=128,
                                          max_iter=3000,
                                          tol=1e-5,
                                          verbose=True)
neural_network_kernel_pca.fit(X_census_kernel_pca, y_census)

# Prediction
prediction_kernel_pca = neural_network_kernel_pca.predict(X_census_kernel_pca)
prediction_kernel_pca_accuracy = accuracy_score(y_census, prediction_kernel_pca)
# LDA
neural_network_lda = MLPClassifier(hidden_layer_sizes=(1, 1),
                                   activation='relu',
                                   solver='adam',
                                   batch_size=128,
                                   max_iter=3000,
                                   tol=1e-5,
                                   verbose=True)
neural_network_lda.fit(X_census_lda, y_census)

# Prediction
prediction_lda = neural_network_lda.predict(X_census_lda)
prediction_lda_accuracy = accuracy_score(y_census, prediction_lda)

# Pos-processing
# PCA
print(f'\nAccuracy PCA: {prediction_pca_accuracy}'
      f'\nClassification Report PCA: '
      f'\n{classification_report(y_census, prediction_pca)}')
cm = ConfusionMatrix(neural_network_pca)
cm.fit(X_census_pca, y_census)
cm.score(X_census_pca, y_census)
cm.show()
# KernelPCA
print(f'\nAccuracy KernelPCA: {prediction_kernel_pca_accuracy}'
      f'\nClassification Report KernelPCA: '
      f'\n{classification_report(y_census, prediction_kernel_pca)}')
cm = ConfusionMatrix(neural_network_kernel_pca)
cm.fit(X_census_kernel_pca, y_census)
cm.score(X_census_kernel_pca, y_census)
cm.show()
# LDA
print(f'\nAccuracy LDA: {prediction_lda_accuracy}'
      f'\nClassification Report LDA: '
      f'\n{classification_report(y_census, prediction_lda)}')
cm = ConfusionMatrix(neural_network_lda)
cm.fit(X_census_lda, y_census)
cm.score(X_census_lda, y_census)
cm.show()
