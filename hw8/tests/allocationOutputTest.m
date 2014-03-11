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
ANN = ArtificialNeuralNetwork(0, 1, 1, 5, absval); % single layer, single node, single input
input = 1;
output = ANN.getOutput(input);
assert( all( output >= 0 ) && all( output <= .5 ) );

ANN = ArtificialNeuralNetwork(1, 4, 4, 5, absval);
input = ones(1,4);
output = ANN.getOutput(input);
assert( all( output >= 0 ) && all( output <= .5*2 ) );

disp('ArtificialNeuralNetwork functional :)');

%% Test training ANN with single node
% Train for sigmoid function, weights shoudl go to 0 and 1

x = -1:.1:1;
y = 1 ./ (1 + exp(-x));
td = [x',y'];
ANN = ArtificialNeuralNetwork(0, 1, 1, 1, sigmoid); % single layer, single node, single input
fprintf('%f != %f\n', ANN.getOutput(1), sigmoid(1));
fprintf('%f != %f\n', ANN.getOutput(-1), sigmoid(-1));
ANN.train(td);
fprintf('%f ~= %f\n', ANN.getOutput(1), sigmoid(1));
fprintf('%f ~= %f\n', ANN.getOutput(-1), sigmoid(-1));
disp('Trained sigmoid');

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
ANN = ArtificialNeuralNetwork(1, 3, 8, 8, perceptron); % single layer, single node, single input
ANN.train(td);
for i = 1:size(x,1)
   e = x(i,:);
   disp('======')
   disp(e)
   disp( ANN.getOutput(e) )
end

disp('Trained identity')      
disp('ArtificialNeuralNetwork trains :)');

