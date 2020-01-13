import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
from matplotlib import rc
from scipy import signal
import imp
import matplotlib as mpl
def downsample_vecs(df, factor=1, plot=True):
    """
    Downsample a vector by a factor 'factor'.
    """
    idx = np.arange(0, len(df), factor)
    dff = df.iloc[idx,:]
    if plot is True:
        plt.figure()
        plt.plot(df.index, df['v'], c='b', label='original')
        plt.scatter(dff.index, dff['v'], c='r', label='downsampled')
        plt.ylabel('Voltage (v)')
        plt.xlabel('Time index')
        plt.legend(loc='upper right')  
        plt.grid()
        plt.show()
    return dff


def plot_volt_profiles(vs, va, t_scale=None, title=None):
    """
    Plot voltage profiles of "secure" and "attack" nodes.
    """
    #plt.close('all')
    if t_scale is None:
        t = np.arange(0, len(vs))
        x_label = 'Time index'
    else:
        t = vs['t'].values * t_scale
        if t_scale//10**9 == 1:
            x_label = 'Time (ns)'
        elif t_scale//10**6 == 1:
            x_label = 'Time ($\mu$s)'
        elif t_scale//10**3 == 1:
            x_label = 'Time (ms)'
        
    plt.figure()
    plt.plot(t, vs['v'], label='secure')
    plt.plot(t, va['v'], label='attack')
    plt.title('Voltage profile {}'.format(title))
    plt.xlabel(x_label)
    plt.ylabel('Voltage (v)')
    plt.legend(loc='upper right')
    plt.xlim((0,200))
    plt.grid()
    plt.show()
    
    
def vec_to_mat(df_vec, n_features=72):
    """
    df: dataframe which contains time and voltage vectors
    n_features: number of temporal samples which are essentially our features
    label: 0 (secured), 1 (under attack)
    """
    v = df_vec['v'] # get the voltage vector
    n_obs = len(v)//n_features # compute the number of observations
    mat = v[:n_obs*n_features].values.reshape(n_obs, n_features) # rashape to matrix size
    df_mat = pd.DataFrame(mat) # generate dataframe
    df_mat['y'] = df_vec['y'][0] # add the label vector to the dataframe
    return df_mat


def combine_datasets(df1, df2, n_features=72):
    """
    
    """
    df_mat1 = vec_to_mat(df1, n_features = n_features)
    df_mat2 = vec_to_mat(df2, n_features = n_features)
    
    # concatenate to dataframes and shuffle the observations
    df = pd.concat([df_mat1, df_mat2]).reset_index(drop=True)
    df = df.reindex( np.random.permutation(df.index) )
    df = df.reset_index(drop=True)
    return df    


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')



class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='1',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Truth label')
    plt.xlabel('Predicted label')
