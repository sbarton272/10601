classdef ArtificialNeuralNetwork
    % ArtificialNeuralNetwork
    % Articicial Neural Network with one hidden layer
    % Uses the sigmoid function as the node threshold function
   
    properties
      % training algorithm
      TrainingAlg = BackPropegation; % TODO implement
        
      % size parameters
      nHiddenLayers = 1;
      nHiddenNodes = 1;
      nInputs = 1;
      nOutputs = 1;
      
      % layers of Nodes
      hiddenLayers;
      outputLayer;
      
      % Various network functions
      nodeThreshFunct = sigmoid;
      outputThreshFunct = @(x) x; %identity function
      
   end
   
   methods
      function obj = ArtificialNeuralNetwork(TrainingAlg, nHiddenLayers, ...
                        nHiddenNodes, nInputs, nOutputs, ...
                        nodeThreshFunct, outputThreshFunct)
          % initialize ANN
          if (nargin > 0)
              obj.TrainingAlg = TrainingAlg;
              obj.nHiddenLayers = nHiddenLayers;
              obj.nHiddenNodes = nHiddenNodes;
              obj.nInputs = nInputs;
              obj.nOutputs = nOutputs;
              obj.nodeThreshFunct = nodeThreshFunct;
              obj.outputThreshFunct = outputThreshFunct;
          end
          
          % init layers vectors
          % init backwards important for pre allocation
          for n = obj.nOutputs:-1:1
            obj.outputLayer(1,n) = ANNNode(obj.nHiddenNodes, ...
                                           obj.nodeThreshFunct);
          end
          
          % hidden layers
          % TODO finish this
          for n = obj.nHiddenNodes:-1:1
            obj.hiddenLayers(1,n) = ANNNode(obj.nHiddenNodes, ...
                                           obj.nodeThreshFunct);
          end
          
      end % ArtificialNeuralNetwork
      
      function output = getOutput(input)
          assert( isEqaul( size(inputs), [1, obj.nInputs] ), ...
            ['ArtificialNeuralNetwork.getOutput:', ...
            'Inputs must be vector of correct size']);
          output = 1:obj.nOutputs; % prealloc
          % TODO finish
          
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