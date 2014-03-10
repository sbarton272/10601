classdef ArtificialNeuralNetwork < handle
    % ArtificialNeuralNetwork
    % Articicial Neural Network with one hidden layer
    % Uses the sigmoid function as the node threshold function
   
    properties (Constant = true)
       sigmoid = @(x) (1/(1 - exp(-x)));
    end
    
    properties
      % training algorithm
      TrainingAlg = []% BackPropegation; % TODO implement
        
      % size parameters
      nLayers = 2; % includes output layer
      nHiddenNodes = 4;
      nInputs = 1;
      nOutputs = 1;
      
      % layers of Nodes
      % output layer is the top layer at index 1
      % hidden layers are ordered from layer 2 down
      layers;
      
      % Various network functions
      nodeThreshFunct = sigmoid;
      outputThreshFunct = @(x) x; %identity function
      
      % flags
      bTrained = false;
      
   end
   
   methods
      function obj = ArtificialNeuralNetwork(TrainingAlg, nHiddenLayers, ...
                        nHiddenNodes, nInputs, nOutputs, ...
                        nodeThreshFunct, outputThreshFunct)
          % initialize ANN
          if (nargin > 0)
              obj.TrainingAlg = TrainingAlg;
              obj.nLayers = nHiddenLayers + 1; % one output layer guaranteed
              obj.nHiddenNodes = nHiddenNodes;
              obj.nInputs = nInputs;
              obj.nOutputs = nOutputs;
              obj.nodeThreshFunct = nodeThreshFunct;
              obj.outputThreshFunct = outputThreshFunct;
          end
          
          % output layer allocation
          obj.layers = 1:obj.nLayers; % pre alloc
          % output layer
          obj.layers(1,1) = ANNNodeLayer(obj.nOutputs, obj.nHiddenNodes,...
                                         obj.nodeThreshFunct);
          % init bottom hidden layer
          obj.layers(1,obj.nLayers) = ANNNodeLayer(obj.nHiddenNodes, ...
                                                   obj.nInputs,...
                                                   obj.nodeThreshFunct);          
          % init internal hidden layers
          for layerN = 2:(obj.nLayers-1)
            obj.layers(1,layerN) = ANNNodeLayer(obj.nHiddenNodes, ...
                                                obj.nHiddenNodes,...
                                                obj.nodeThreshFunct);
          end
          
      end % ArtificialNeuralNetwork
      
      function output = getOutput(obj, input)
          % getOutput Get the output vector of a trained ANN given an input
          % vector
          assert( isEqaul( size(inputs), [1, obj.nInputs] ), ...
            ['ArtificialNeuralNetwork.getOutput:', ...
            'Inputs must be vector of correct size']);
                    
          % propogate input through the layers with each passing new input
          % forward
          for layerN = 1:obj.nHiddenLayers
              input = obj.layers(1,layerN).getLayerOutput(input);
              % TODO may be horribly inefficient
          end
          
          % output from final layer is the result
          output = input;
          
      end % getOutput
      
      function train(obj)
          obj.TrainingAlg.train();
          obj.bTrained = true;
      end % train
      
   end % methods
end % classdef