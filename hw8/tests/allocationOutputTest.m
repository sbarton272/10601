%% Test the allocation and output features of the ANN

clear;
addpath('..');

absval = @(x) abs(x);
identity = @(x) x;
negFunct = @(x) -x;
sigmoid = @(x) (1/(1 + exp(-x)));

%% test basic node
node = ANNNode(2, negFunct);

newWeight = [.5, 0];
node.updateWeights(newWeight);
assert( isequal( node.weights, newWeight ) );

input = [-2, 1];
output = node.getOutput(input);
assert( isequal( output, 1 ) );
disp('ANNNode functional :)');

%% Test basic node layer
nodeLayer = ANNNodeLayer(5, 4, sigmoid);
sumWeights = sum(nodeLayer.getLayerOutput( ones(1,4) )); % all output [0,1] 
assert( sumWeights <= 5 );
disp('ANNNodeLayer functional :)');


%% Test basic Artificial Neural Network
ANN = ArtificialNeuralNetwork(0, 1, 1, 5,identity, absval); % single layer, single node, single input
input = 1;
output = ANN.getOutput(input);
assert( all( output >= 0 ) && all( output <= .5 ) );

ANN = ArtificialNeuralNetwork(1, 4, 4, 5,identity, absval);
input = ones(1,4);
output = ANN.getOutput(input);
assert( all( output >= 0 ) && all( output <= .5*2 ) );

disp('ArtificialNeuralNetwork functional :)');

%% Test training ANN with single node
% Train for identity function

td = [1 1; -1 0];
ANN = ArtificialNeuralNetwork(1, 1, 1, 1,sigmoid, sigmoid); % single layer, single node, single input
out1 = ANN.getOutput(1)
out_1 = ANN.getOutput(-1)
ANN.train(td)
out1 = ANN.getOutput(1)
out_1 = ANN.getOutput(-1)

disp('ArtificialNeuralNetwork trains :)');

