%% Test the allocation and output features of the ANN

clear;
addpath('..');

absval = @(x) abs(x);
identity = @(x) x;
negFunct = @(x) -x;
sigmoid = @(x) (1/(1 + exp(-x)));
perceptron = @(x) x >= .5;

%% test basic node
node = ANNNode(2, negFunct);

newWeight = [0, .5, 0];
node.setWeights(newWeight);
node.weights
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
ANN = ArtificialNeuralNetwork(0, 1, 1, 1); % single layer, single node, single input
input = 1;
output = ANN.getOutput(input);
assert( output >= 0 && output <= 1 );

ANN = ArtificialNeuralNetwork(1, 4, 4, 5);
input = ones(1,4);
output = ANN.getOutput(input);
assert( all( output >= 0 ) && all( output <= 1 ) );

disp('ArtificialNeuralNetwork functional :)');

%% Test training ANN with single node
% Train for sigmoid function, weights shoudl go to 0 and 1

x = -1:.1:1;
y = 1 ./ (1 + exp(-x));
td = [x',y'];
ANN = ArtificialNeuralNetwork(0, 1, 1, 1); % single layer, single node, single input
fprintf('%f != %f\n', ANN.getOutput(1), sigmoid(1));
fprintf('%f != %f\n', ANN.getOutput(-1), sigmoid(-1));
ANN.train(td);
fprintf('%f ~= %f\n', ANN.getOutput(1), sigmoid(1));
fprintf('%f ~= %f\n', ANN.getOutput(-1), sigmoid(-1));
disp('Trained sigmoid');

% test smaller identiy
x = [ 0 1;
      1 0];
y = x;
td = [x,y];
ANN = ArtificialNeuralNetwork(1, 1, 2, 2); % single layer, single node, single input
% set weights for easier debugging
ANN.layers(1).nodes(1).setWeights([-.5 1]);
ANN.layers(1).nodes(2).setWeights([.5 -1]);
ANN.layers(2).nodes(1).setWeights([.001 1 -1]);

% disp(ANN.layers(1).nodes(1).weights);
% disp(ANN.layers(1).nodes(2).weights);
% disp(ANN.layers(2).nodes(1).weights);

for i = 1:1000
    ANN.train(td);
%     disp('--------------');
%     disp(ANN.layers(1).nodes(1).weights);
%     disp(ANN.layers(1).nodes(2).weights);
%     disp(ANN.layers(2).nodes(1).weights);
end

for i = 1:size(x,1)
   e = x(i,:);
   disp('======')
   disp(e)
   disp( ANN.getOutput(e) > .5 )
end

% test identiy
x = [ 0 0 0 0 0 0 0 1;
      0 0 0 0 0 0 1 0;
      0 0 0 0 0 1 0 0;
      0 0 0 0 1 0 0 0;
      0 0 0 1 0 0 0 0;
      0 0 1 0 0 0 0 0;
      0 1 0 0 0 0 0 0;
      1 0 0 0 0 0 0 0];
y = x;
td = [x,y];
ANN = ArtificialNeuralNetwork(1, 3, 8, 8); % single layer, single node, single input

for i = 1:10000
    ANN.train(td);
end

for i = 1:size(x,1)
   e = x(i,:);
   disp('======')
   disp(e)
   disp( ANN.getOutput(e) > .5 )
end

disp('Trained identity')      
disp('ArtificialNeuralNetwork trains :)');

