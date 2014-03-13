%% Spencer Barton
% 10601
% NN_education.m

function NN_education(trainingFileName, testFileName)

%% load files 
trainFID = fopen(trainingFileName,'rt');
testFID = fopen(testFileName,'rt');

% cell arrays
trainingDataCell = textscan(trainFID, '%s %s %s %s %s %s', 'Delimiter', ',', 'CollectOutput', 1);
testDataCell = textscan(testFID, '%s %s %s %s %s', 'Delimiter', ',', 'CollectOutput', 1);

trainingDataCell = trainingDataCell{1}(2:end,:);
testDataCell = testDataCell{1}(2:end,:);

fclose(trainFID);
fclose(testFID);

%% collect input and output vectors

% convert from cell and convert 0-100 to 0-1 range
trainingData = str2double(trainingDataCell) / 100;
testingData = str2double(testDataCell) / 100;

%% create ANN and train
% Print squared error every 10 iterations
ANN = ArtificialNeuralNetwork(1,5,5,1,1,10);

% set weights based on prior iteration
load('educationWeights.mat', 'weights');
ANN.setAllWeights(weights);

target = trainingData(:,6);
inputs = trainingData(:,1:5);
for iterN = 30:-1:1
    % display error for all inputs
    outputs = ANN.getOutput(inputs);
    errs(iterN) = sqrErr(outputs, target);
    fprintf( '%f\n', errs(iterN) );
    % train for 10 iterations
    ANN.train(trainingData);
end

% save weights if they are good
weights = ANN.getAllWeights();
save('educationWeights.mat', 'weights');

fprintf('TRAINING COMPLETED! NOW PREDICTING.\n');
plot(30:-1:1,errs)

%% predict the test data
prediction = ANN.getOutput(testingData);

for i = 1:length(prediction)
    fprintf('%.1f\n', prediction(i)*100);
end
    
end

%% Helper functions

function err = sqrErr(val, target)
    err = sum(.5*(val - target).^2);
end