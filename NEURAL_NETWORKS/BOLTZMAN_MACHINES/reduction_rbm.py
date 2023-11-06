import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
predictors = np.asarray(base.data, 'float32')
classes = base.target

norm = MinMaxScaler(feature_range=(0,1))
predictors = norm.fit_transform(predictors)

predictors_train, predictors_test, classes_train, classes_test = train_test_split(predictors, classes, test_size=0.2, random_state=0)

rbm = BernoulliRBM(random_state = 0, n_iter=25, n_components=50)
naive_rbm = GaussianNB()

classificator_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
classificator_rbm.fit(predictors_train, classes_train)

plt.figure(figsize=(20,20))
for  i, comp in enumerate(rbm.components_):
    plt.subplot(10,10,i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

predictions_rbm = classificator_rbm.predict(predictors_test)
precision_rbm = metrics.accuracy_score(predictions_rbm, classes_test)
# precision = 0.886111

simple_naive = GaussianNB()
simple_naive.fit(predictors_train, classes_train)
predictions_naive = simple_naive.predict(predictors_test)
precision_naive = metrics.accuracy_score(predictions_naive, classes_test)
# precision = 0.81111

# reduction through rbm improves accuracy