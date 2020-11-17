# ----------------------------------------------------------------------------
# Import
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from ML_Function import *
# ----------------------------------------------------------------------------
# Streamlit

st.title("Streamlit Example")

st.write("""
# Explore Different Classifier
Which on is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

st.write("Dataset selected by user: {}".format(dataset_name))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else : 
        data = datasets.load_wine()

    X = data.data 
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of Dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))



params = add_parameter_ui(classifier_name)



clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write("Classifier: {}".format(classifier_name))
st.write("Accuracy = {}".format(acc))

# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)


x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)