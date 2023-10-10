
%Step 1: Import data.

traindata = readtable('F:/Amino acid-classifier/classifier/training_set_20aa.xlsx');%Import trainning dataset
testdata = readtable('F:/Amino acid-classifier/classifier/testing_set_20aa.xlsx');%Import testing dataset
predictdata = readtable('F:/Amino acid-classifier/classifier/predicting_set_20aa.xlsx');%Import predicting dataset

% Step 2: Train quadratic SVM Classifier.

% Extract predictors and response.
% This code processes the data into the right shape for training the model.
predictorNames = {'mean', 'std', 'skew', 'kurt', 'toff'};
predictors = traindata(:, predictorNames);
response = traindata.label;
isCategoricalPredictor = [false, false, false, false, false];

% Train a classifier.
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical({'A'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'K'; 'L'; 'M'; 'N'; 'P'; 'Q'; 'R'; 'S'; 'T'; 'V'; 'W'; 'Y'}));

% Create the result struct with predict function.
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct.
trainedClassifier.RequiredVariables = {'kurt', 'mean', 'skew', 'std', 'toff'};
trainedClassifier.ClassificationSVM = classificationSVM;

% Extract predictors and response.
% This code processes the data into the right shape for training the model.
predictorNames = {'mean', 'std', 'skew', 'kurt', 'toff'};
predictors = traindata(:, predictorNames);
response = traindata.label;
isCategoricalPredictor = [false, false, false, false, false];

% Perform cross-validation.
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

% Compute validation predictions.
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy.
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

% Compute testing accuracy
predLetter = predict(trainedClassifier.ClassificationSVM,testdata);
testingAccuracy = 1-sum(predLetter ~= testdata.label)/numel(predLetter)

% Export confusion matrix.
validationPredictions_cellstr = cellstr(validationPredictions);
subplot(2,2,1)
cm = confusionchart(response,validationPredictions_cellstr, ...
    'Title','Confusion Matrix');
xlabel('Predicted class');ylabel('True class')

%  maximizing window of figure
scrsz = get(0,'ScreenSize');
set(gcf,'Position',scrsz);

%STEP 3: Make prediction with the returned 'trainedClassifier' on predicting set.

labelpredict = predict(trainedClassifier.ClassificationSVM,predictdata);

% Export scatter plot with labels
mean = table2array(predictdata(:,1));
std = table2array(predictdata(:,2));
newcolors = [4	0	0
    231	33	26
    24	73	158
    171	84	156
    24	124	58
    30	45	91
    89	69	153
    118	28	120
    123	23	28
    125	126	45
    43	95	153
    36	144	144
    150	99	37
    66	177	70
    174	39	115
    54	185	191
    60	62	143
    143	166	47
    127 128 128
    148	99	99];
subplot('position',[.55  .175  .4  .7])
gscatter(mean,std,labelpredict,newcolors/255)
xlim([30,105]);xlabel('mean/pA');ylabel('std/pA');title('Predicted Labels')
legend('location','bestoutside')

%STEP 4: Plot learning curve.
warning('off')
a = height(traindata);
k = 0;
for i = 200 : 100 : a;
    sample = traindata(randi(a,1,i),:);
    k = k + 1;
    s(k) = i;
    predictorNames = {'mean', 'std', 'skew', 'kurt', 'toff'};
    predictors = sample(:, predictorNames);
    response = sample.label;
    isCategoricalPredictor = [false, false, false, false, false];
    template = templateSVM(...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder', 2, ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    classificationSVM = fitcecoc(...
        predictors, ...
        response, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', categorical({'A'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'K'; 'L'; 'M'; 'N'; 'P'; 'Q'; 'R'; 'S'; 'T'; 'V'; 'W'; 'Y'}));
    
    predictorExtractionFcn = @(t) t(:, predictorNames);
    ensemblePredictFcn = @(x) predict(classificationSVM, x);
    trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
    
    trainedClassifier.RequiredVariables = {'mean', 'std', 'skew', 'kurt', 'toff'};
    trainedClassifier.ClassificationSVM = classificationSVM;
    
    predictorNames = {'mean', 'std', 'skew', 'kurt', 'toff'};
    predictors = sample(:, predictorNames);
    response = sample.label;
    isCategoricalPredictor = [false, false, false, false, false];
    
    % Perform cross-validation.
    partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);
    
    % Compute validation predictions.
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
    
    % Compute validation accuracy.
    validationAccuracy (k) = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
    
    % Compute resubstitution accuracy
    resubstitutionAccuracy (k) = 1 - resubLoss(trainedClassifier.ClassificationSVM);
end
subplot(2,2,3)
plot(s,resubstitutionAccuracy,'-k*','linewidth',2);hold on;
plot(s,validationAccuracy,'-r*','linewidth',2);
ylim([0.90,1.01]);xlabel('Training sample');ylabel('Accuracy');
title('Learning curve')
legend('Training','Validation','location','best')

%output plots
saveas(gcf,'output.jpg') 
