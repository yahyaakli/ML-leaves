clear all
close all
clc

digitDatasetPath = fullfile('.\Leaf\');
imds =imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

figure;
perm = randperm(620,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

%% label count
labelCount = countEachLabel(imds);

%% image size
img = readimage(imds,1);
size(img)

%% devide elements
numTrainFiles = 120;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% train network 
layers = [
    imageInputLayer([175 175 3])
    
    convolution2dLayer(2,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

%% compute validation accuracy

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);


