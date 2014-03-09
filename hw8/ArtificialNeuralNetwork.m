classdef ArtificialNeuralNetwork
    % ArtificialNeuralNetwork
    % Articicial Neural Network with one hidden layer
    % Uses the sigmoid function as the node threshold function
   
    properties (SetAccess = private)
      % training algorithm
      TrainingAlg;
        
      % size parameters
      nHiddenLayers = 1;
      nHiddenNodes;
      nInputs;
      nOutputs;
      
      % layers of Nodes
      hiddenLayers;
      outputLayer;
      
      % Various network functions
      nodeThreshFunct = sigmoid;
      outputThreshFunct;
      
   end
   
   methods
      function obj = ArtificialNeuralNetwork(nHiddenLayers, nInputs)
          % initialize ANN with the number
          if nargin < 2)
          end
      end % ArtificialNeuralNetwork
      
      function getOutput(input)
          
      end % getOutput
      
      function train()
          obj.TrainingAlg.train();
      end % train
      
      function y = sigmoid(x)
        % sigmoid function with basic unit parameters
        y = 1 / ( 1 + exp(-x) );
      end % sigmoid
      
   end % methods
end % classdef