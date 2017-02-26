# import all necessary libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load the dataset (local path)
url = "datasets/parkinsons_data.txt"
# feature names
features = ["name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1","spread2","D2","PPE"]
dataset = pandas.read_csv(url, names = features)

# dataset dimensions (should be (150,5) --> 150 instances, 5 attributes)
print(dataset.shape)
# first 20 instances (rows) of the data)
print(dataset.head(20))
# statistical summary of the dataset (mean, min, max, count, std, etc)
print(dataset.describe())
# class distrubution (number of instances that belong to each class of flower)
print(dataset.groupby('class').size())

# box-and-whisker univariate plots (plots of each individual variable)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('box-and-whisker-plots.png', dpi=100)
plt.show()
# histogram univariate plots
dataset.hist()
plt.savefig('histograms.png', dpi=100)
plt.show()
# scatter plot matrix multivariate plot
scatter_matrix(dataset)
plt.savefig('scatter-matrix.png', dpi=100)
plt.show()

# store the dataset as an array for easier processing
array = dataset.values
# X stores feature values (every row, 0 through 3rd column)
X = array[:,0:4]
# Y stores "answers", the flower species / class (every row, 4th column)
Y = array[:,4]
validation_size = 0.20
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (80%) and validation set (20%)
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'

# algorithms / models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each algorithm / model
results = []
names = []
print("Scores for each algorithm:")
for name, model in models:
    kfold = cross_validation.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print(name, ":", 100.0*cv_results.mean(), "%")

# plot algorithm comparison (boxplot)
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig('Algorithm-comparison.png')
plt.show()

# using KNN to make predictions about the validation set
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
