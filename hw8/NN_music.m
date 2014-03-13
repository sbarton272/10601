%% Spencer Barton
% 10601
% NN_music.m

function NN_music(trainingFileName, testFileName)

%% load files 
trainFID = fopen(trainingFileName,'rt');
testFID = fopen(testFileName,'rt');

% cell arrays
trainingDataCell = textscan(trainFID, '%s %s %s %s %s', 'Delimiter', ',', 'CollectOutput', 1);
testDataCell = textscan(testFID, '%s %s %s %s', 'Delimiter', ',', 'CollectOutput', 1);

trainingDataCell = trainingDataCell{1}(2:end,:);
testDataCell = testDataCell{1}(2:end,:);

fclose(trainFID);
fclose(testFID);

%% collect input and output vectors

trainingData = zeros(length(trainingDataCell),5);
testingData = zeros(length(testDataCell),4);

% convert years from 1900-2000 to range from 0-1
trainingData(:,1) = (str2double(trainingDataCell(:,1)) - 1900) / 100;
testingData(:,1) = (str2double(testDataCell(:,1)) - 1900) / 100;

% convert duration from range 0-7 to range from 0-1
trainingData(:,2) = str2double(trainingDataCell(:,2)) / 7;
testingData(:,2) = str2double(testDataCell(:,2)) / 7;

% convert yes/no to 1/-1 for Jazz, Rock and Hit?
for row = 1:length(trainingDataCell)
    trainingData(row,3) = convertFromYN(trainingDataCell(row,3));
    trainingData(row,4) = convertFromYN(trainingDataCell(row,4));
    trainingData(row,5) = convertFromYN(trainingDataCell(row,5));
end

for row = 1:length(testDataCell)
    testingData(row,3) = convertFromYN(testDataCell(row,3));
    testingData(row,4) = convertFromYN(testDataCell(row,4));    
end

%% create ANN and train
% Print squared error every 10 iterations
ANN = ArtificialNeuralNetwork(1,4,4,1,1,10);

% set weights based on prior iteration
load('musicWeights.mat', 'weights');
ANN.setAllWeights(weights);


target = trainingData(:,5);
inputs = trainingData(:,1:4);
for iterN = 30:-1:1
    % display error for all inputs
    outputs = ANN.getOutput(inputs);
    errs(iterN) = sqrErr(outputs, target);
    fprintf( '%.2f\n', errs(iterN) );
    % train for 10 iterations
    ANN.train(trainingData);
end

% save weights if they are good
% weights = ANN.getAllWeights();
% save('musicWeights.mat', 'weights');

fprintf('TRAINING COMPLETED! NOW PREDICTING.\n');
% plot(30:-1:1,errs)

%% predict the test data
prediction = ANN.getOutput(testingData);

for i = 1:length(prediction)
    fprintf('%s\n', convertToYN(prediction(i)));
end
    
end

%% Helper functions

function err = sqrErr(val, target)
    err = sum(.5*(val - target).^2);
end

function out = convertFromYN(y_n)
    if strcmp(y_n,'yes')
        out = 1;
    elseif strcmp(y_n,'no')
        out = -1;
    end
end

function out = convertToYN(y_n)
    if y_n >= .5
        out = 'yes';
    else
        out = 'no';
    end
end