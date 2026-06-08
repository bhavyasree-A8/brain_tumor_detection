-%% Brain Tumor Detection using ResNet50

datasetPath = 'brain_tumor_dataset';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

countEachLabel(imds)


%% Split Dataset

[trainImgs,testImgs] = splitEachLabel(imds,0.7,'randomized');


%% Resize Images

inputSize = [224 224];

augTrain = augmentedImageDatastore(inputSize,trainImgs,'ColorPreprocessing','gray2rgb');
augTest = augmentedImageDatastore(inputSize,testImgs,'ColorPreprocessing','gray2rgb');


%% Load ResNet50

net = resnet50;
lgraph = layerGraph(net);

numClasses = numel(categories(trainImgs.Labels));

newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc_new')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayers(3));


%% Training Options

options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{augTest,testImgs.Labels}, ...
    'Verbose',false, ...
    'Plots','training-progress');


%% Train Model

trainedNet = trainNetwork(augTrain,lgraph,options);


%% Accuracy

predLabels = classify(trainedNet,augTest);
accuracy = mean(predLabels == testImgs.Labels)


%% Confusion Matrix

figure
confusionchart(testImgs.Labels,predLabels)

%% Test on New MRI Image

img = imread('tumor.png');   % use any MRI image
img_resized = imresize(img,[224 224]);

label = classify(trainedNet,img_resized);

figure
imshow(img)
title(['Prediction: ',char(label)])

%% Tumor Region Highlighting

gray = rgb2gray(img);
gray = imadjust(gray);

bw = imbinarize(gray,0.7);
bw = bwareaopen(bw,150);

stats = regionprops(bw,'BoundingBox','Area','Centroid');

bestIdx = -1;
bestArea = 0;

for i = 1:length(stats)
    x = stats(i).Centroid(1);
    y = stats(i).Centroid(2);
    
    if x > 50 && x < size(gray,2)-50 && y > 50 && y < size(gray,1)-50
        if stats(i).Area > bestArea
            bestArea = stats(i).Area;
            bestIdx = i;
        end
    end
end

figure
imshow(img)
hold on

if bestIdx > 0
    rectangle('Position',stats(bestIdx).BoundingBox,'EdgeColor','r','LineWidth',3);
end

title('Detected Tumor Region')