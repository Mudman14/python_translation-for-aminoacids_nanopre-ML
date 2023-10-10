import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


traindata = pd.read_excel('training_set_20aa.xlsx')
testdata = pd.read_excel('testing_set_20aa.xlsx')
predictdata = pd.read_excel('predicting_set_20aa.xlsx')


predictorNames = ['mean', 'std', 'skew', 'kurt', 'toff']
predictors = traindata[predictorNames]
response = traindata['label']
le = LabelEncoder()
response = le.fit_transform(response)


svm = SVC(kernel='poly', degree=2, C=1, decision_function_shape='ovo')
classificationSVM = svm.fit(predictors, response)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classificationSVM, predictors, response, cv=10)
validationAccuracy = np.mean(scores)


testPredictions = classificationSVM.predict(testdata[predictorNames])
testAccuracy = np.mean(testPredictions == le.transform(testdata['label']))


cm = confusion_matrix(le.transform(testdata['label']), testPredictions)
plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
plt.xlabel('Predicted class')
plt.ylabel('True class')


predictions = classificationSVM.predict(predictdata[predictorNames])


plt.figure()
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i, label in enumerate(le.classes_):
    plt.scatter(predictdata['mean'][predictions == i], predictdata['std'][predictions == i], c=colors[i], label=label)
plt.xlabel('mean/pA')
plt.ylabel('std/pA')
plt.title('Predicted Labels')
plt.legend()


resubstitutionAccuracy = []
validationAccuracy = []
for i in range(200, len(traindata), 100):
    sample = traindata.sample(n=i)
    predictors = sample[predictorNames]
    response = sample['label']
    response = le.transform(response)
    svm = SVC(kernel='poly', degree=2, C=1, decision_function_shape='ovo')
    classificationSVM = svm.fit(predictors, response)
    scores = cross_val_score(classificationSVM, predictors, response, cv=10)
    validationAccuracy.append(np.mean(scores))
    resubstitutionAccuracy.append(classificationSVM.score(predictors, response))

plt.figure()
plt.plot(range(200, len(traindata), 100), resubstitutionAccuracy, '-k*', linewidth=2, label='Training')
plt.plot(range(200, len(traindata), 100), validationAccuracy, '-r*', linewidth=2, label='Validation')
plt.ylim([0.90, 1.01])
plt.xlabel('Training sample')
plt.ylabel('Accuracy')
plt.title('Learning curve')
plt.legend()


plt.savefig('output.jpg')