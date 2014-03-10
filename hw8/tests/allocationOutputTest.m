%% Test the allocation and output features of the ANN

clear;

negFunct = @(x) -x;

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


%% Test basic Artificial Neural Network
ANN = ArtificialNeuralNetwork([], 0, 0, 1, 1,[], []);