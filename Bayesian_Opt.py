import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.base import clone
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn import datasets, svm
from sklearn.svm import LinearSVR
from scipy.stats import uniform
from xgboost import XGBRegressor
'''
Diabetes
'''
# Load the diabetes dataset (for regression)
X, Y = datasets.load_diabetes(return_X_y=True)

########### XGBRegressor
# Instantiate an XGBRegressor with default hyperparameter settings
xgb = XGBRegressor()

# and compute a baseline to beat with hyperparameter optimization
baseline = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error').mean()

# Hyperparameters to tune and their ranges
param_dist = {"learning_rate": uniform(0, 1),
              "gamma": uniform(0, 5),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10)}

rs = RandomizedSearchCV(xgb, param_distributions=param_dist,
                        scoring='neg_mean_squared_error',
                        n_iter=25, cv = 5,
                        n_jobs = -1,
                        verbose = 1)

# Run random search for 25 iterations
rs.fit(X, Y)

bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]

# Optimization objective
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                XGBRegressor(learning_rate=parameters[0],
                              gamma=int(parameters[1]),
                              max_depth=int(parameters[2]),
                              n_estimators=int(parameters[3]),
                              min_child_weight = parameters[4]),
                X, Y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score

optimizer = BayesianOptimization(f=cv_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=20)

y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

plt.plot(-y_rs, 'ro-', label='Random search')
plt.plot(-y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.ylim(3000,5000)
plt.title('Value of the best sampled CV score')
plt.legend()


############ linear SVM
SVM = LinearSVR(random_state=0, tol=1e-5)
# and compute a baseline to beat with hyperparameter optimization
# baseline = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error').mean()
distributions = dict(C=uniform(loc=0.0001, scale=1000))
clf = RandomizedSearchCV(SVM, distributions, scoring='neg_mean_squared_error',
                        n_iter=25, cv = 5,
                        n_jobs = -1,
                        verbose = 1)

# Run random search for 25 iterations
rs_svm = clf.fit(X, Y)
bds_svm = [{'name': 'C', 'type': 'continuous', 'domain': (0.0001, 1000)}]

# Optimization objective
def cv_score_svm(parameters):
    parameters = parameters[0]
    score = cross_val_score(
        LinearSVR(C=parameters[0], tol=1e-5),
        X, Y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score

optimizer_svm = BayesianOptimization(f=cv_score_svm,
                                 domain=bds_svm,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer_svm.run_optimization(max_iter=20)

y_rs = np.maximum.accumulate(rs_svm.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer_svm.Y).ravel()

print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

plt.plot(-y_rs, 'ro-', label='Random search')
plt.plot(-y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.ylim(3020, 3050)
plt.title('Value of the best sampled CV score')
plt.legend()
'''
MNIST
'''
###### Simple CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 2000, stratify = y_train)
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
n_input = X_train.shape[1:]
# define a simple CNN model
def keras_cnn_model(dense_layers_dim = 128, dropout_rate = 0.5,
                    learning_rate=0.001,  optimizer = 'adam',
                     n_input = n_input, n_class = 10):
  # create model
  model = Sequential()
  model.add(Conv2D(32, (5, 5), input_shape=n_input, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(dropout_rate))
  model.add(Flatten())
  model.add(Dense(n_class, activation='softmax'))
  adam = optimizers.Adam(lr=learning_rate)
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model

# specify other extra parameters pass to the .fit
# number of epochs is set to a large number, we'll
# let early stopping terminate the training process
early_stop = EarlyStopping(
    monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 0, mode='min')

callbacks = [early_stop]
keras_fit_params = {
    'callbacks': callbacks,
    'epochs': 200,
    # 'validation_split': 0.2
    #'validation_data': (X_val, y_val),
    'verbose': 0
}

# sklearn keras wrapper
model_keras = KerasClassifier(
    build_fn = keras_cnn_model,
    n_input = n_input,
    n_class = num_classes

    )

# random search's parameter:
# specify the options and store them inside the dictionary
batch_size_opts = range(16,512) #[128,64,32]
learning_rate_opts = uniform(0.0001,0.01-0.0001)
epochs_opts = range(30, 150)
keras_param_options = {
    'learning_rate': learning_rate_opts,
    'epochs': epochs_opts,
    'batch_size': batch_size_opts
}
# build the randomize searh object
rs_keras = RandomizedSearchCV(
    model_keras,
    param_distributions = keras_param_options,
    # fit_params = keras_fit_params,
    scoring = 'neg_log_loss',
    n_iter = 10,
    cv = 4,
    n_jobs = 1,
    verbose = 1
)
# Run random search for 15 iterations
rs_keras.fit(X_train, y_train)

bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 0.01)},
        {'name': 'epochs', 'type': 'discrete', 'domain': (30, 150)}]

# Optimization objective
def cv_score(parameters):
    parameters = parameters[0]
    # fit_params ={'learning_rate': parameters[0],'epochs': int(parameters[1])}
    skf = StratifiedKFold(n_splits=4)
    score = .0
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train.argmax(1))):
        x_train_kf, x_val_kf = X_train[train_index], X_train[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        model = keras_cnn_model(learning_rate=parameters[0])
        model.fit(x_train_kf, y_train_kf, epochs=int(parameters[1]))
        score -= model.evaluate(x_val_kf, y_val_kf)[0]
    return score/4

optimizer = BayesianOptimization(f=cv_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=5)

y_rs = np.maximum.accumulate(rs_keras.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

# print(f'Baseline neg. MSE = {baseline:.2f}')
print(f'Random search neg. MSE = {y_rs[-1]:.5f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.5f}')

plt.plot(-y_rs, 'ro-', label='Random search')
plt.plot(-y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('log loss')
plt.ylim(0.24,0.4)
plt.title('Value of the best sampled CV score')
plt.legend()
plt.show()

###### SVM

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 2000, stratify = y_train)

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28* 28)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

SVM = svm.SVC(random_state=0, tol=1e-5,probability=True)
# and compute a baseline to beat with hyperparameter optimization
# baseline = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error').mean()
distributions = dict(C=uniform(loc=0.0001, scale=1000),gamma=uniform(loc=0, scale=5))
clf = RandomizedSearchCV(SVM, distributions, scoring='neg_log_loss',
                        n_iter=25, cv = 5,
                        n_jobs = -1,
                        verbose = 3)
rs_svm = clf.fit(X_train, y_train);

bds_svm = [{'name': 'C', 'type': 'continuous', 'domain': (0.0001, 1000)},
           {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)}]

# Optimization objective
def cv_score_svm(parameters):
    parameters = parameters[0]
    score = cross_val_score(
        svm.SVC(C=parameters[0],gamma=parameters[1], tol=1e-5,probability=True),
        X_train, y_train, scoring='neg_log_loss').mean()
    score = np.array(score)
    return score

optimizer_svm = BayesianOptimization(f=cv_score_svm,
                                 domain=bds_svm,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer_svm.run_optimization(max_iter=5)


y_rs = np.maximum.accumulate(rs_svm.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer_svm.Y).ravel()

print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

plt.plot(-y_rs, 'ro-', label='Random search')
plt.plot(-y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('log loss')
plt.ylim(0.23, 0.5)
plt.title('Value of the best sampled CV score')
plt.legend()