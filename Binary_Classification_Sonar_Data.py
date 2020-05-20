#Binary Classification of Sonar Data


# 1. Prepare Problem
# a) Load Libraries 
# Load all the required modules and libraries that will be used for our problem.
import numpy
from matplotlib import pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier


# In[2]:


# b) Load Dataset
# Load tha dataset from file/url and this is the place to reduce the sample of dataset specially if it's too large to work with.
# We can always scale up the well performing models later.

filename = 'sonar_data.csv'
dataset = read_csv(filename, header=None)


# In[3]:


# 2. Summarize Data
# This step is to learn more about the dataset which will help us decide which algorithms to use with this data. 
# a) Descriptive Statistics
#shape
print(dataset.shape)
#types
set_option('display.max_rows',500)
print(dataset.dtypes)
#head
set_option('display.width',100)
print(dataset.head(5))
#descrption and change precision to 3 places
set_option('precision', 3)
print(dataset.describe())
#class distribution
print(dataset.groupby(60).size())


# In[4]:


# b) Data Visualizations
#histogram
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
#density plots
dataset.plot(kind='density', subplots=True, sharex=False, legend=False, layout=(8,8), fontsize=1)
pyplot.show()
#many distributions have skewed distribution, A power transform like Box-Cox transform can be useful

#correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()


# In[5]:


# 3. Prepare Data
# a) Data Cleaning - Removing duplicates, dealing with the missing values
# b) Feature Selection
# c) Data Transforms - Scaling/standarizaion of data


# In[9]:


# 4. Evaluate Algorithm
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# b) Test options and evaluate metric
num_fold = 10
scoring = 'accuracy'

# c) Spot Check Algorithms
#baseline Algo
models = []
models.append(('LR',LogisticRegression(solver='liblinear')))
models.append(('LDS',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

#compare Baseline Algos
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s:%f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#LDS shows highest mean accuracy here, let's look at the distribution of accuracy
fig = pyplot.figure()
fig.suptitle('Baseline Algo Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
#LDS is showing the best distribution

#repeating the Algos but with Standardized data
#standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scalar', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scalar', StandardScaler()),('LDS',LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scalar', StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scalar', StandardScaler()),('CART',DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scalar', StandardScaler()),('NB',GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scalar', StandardScaler()),('SVM',SVC(gamma='auto'))])))
#compare
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s:%f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
#after standardization both KNN and SVM improved and now SVM is the best performing one. Plotting for better understanding
fig = pyplot.figure()
fig.suptitle('Scaled Baseline Algo Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# d) Compare Algorithms


# In[10]:


# 5. Improve Accuracy
# a) Algorithm Tuning
#tunning Scaled KNN
print('---------------------------------------------------------------------------------------------------------------------')
print("Tunning of Scaled KNN")
print('---------------------------------------------------------------------------------------------------------------------')

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('Best: %f using %s' %(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))

print('---------------------------------------------------------------------------------------------------------------------')

print("Tunning of Scaled SVM")

print('---------------------------------------------------------------------------------------------------------------------')
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear',  'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('Best: %f using %s' %(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))
print('---------------------------------------------------------------------------------------------------------------------')

#svm is showing the best accuracy so far, which can be improved with ensembles
# b) Ensembles
print('Ensembles with svm')
print('---------------------------------------------------------------------------------------------------------------------')

ensembles =[]
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s:%f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
#compare ensemble algo's
fig = pyplot.figure()
fig.suptitle('Ensemble Algo Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[11]:


# 6. Finalize Model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=1.5)
model.fit(rescaledX, Y_train)
# a) Predictions on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

