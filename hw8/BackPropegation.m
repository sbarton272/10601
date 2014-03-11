classdef BackPropegation < handle
    % BackPropegation Algorithm to train an ANN
    % Relies on the implementation of the ANN
    
    properties
        ANN;
        maxIter = 100;
        learningRate = 1;
        nLayers;
        
        % intermediary storage elements
        nodeOutputs; % cell array ordered by layer (1 - top, bottom is network input)
        nodeErrors; % cell array ordered by layer (1 - top)
        trainingData; % cell array Nx2: data, target
        nTrainingElems;
    end
    
    methods
        function obj = BackPropegation(ANN, learningRate)
            if (nargin > 0)
                obj.ANN = ANN;
                obj.nLayers = obj.ANN.getNLayers();
                obj.learningRate = learningRate;
                
                % init intermediary storage elements, row vects
                obj.nodeOutputs = cell(1, obj.nLayers + 1); % +1 to store input to network as well
                obj.nodeErrors  = cell(1, obj.nLayers); 
            end
        end % BackPropegation
        
        function train(obj, trainingData)
            obj.trainingData = trainingData;
            obj.nTrainingElems = size(trainingData,1);
            
            for iter = 1:obj.maxIter % termination
                % iterate through a set number of times performing gradient
                % decent
                % fprintf('====== %d\n', iter)
                
                for exampleIndex = 1:obj.nTrainingElems
                    % iter through each example performing gradient decent
                    % on each eaxample
                    trainingExample = obj.trainingData(exampleIndex,:);
                    example = trainingExample( 1:obj.ANN.nInputs );
                    exampleTarget = trainingExample( obj.ANN.nInputs+1:end );
                    
                    % fprintf('> %d -> %d\n', example, exampleTarget)
                    
                    % destructively update ANN
                    obj.calculateNodeOutputs(example);
                    obj.calculateNodeErrors(exampleTarget);
                    obj.updateWeights();
                
                end % training examples loop
            end % gradient decent loop
        end % train
                
        function calculateNodeOutputs(obj, input)
            % propegate the training example input and store the
            % intermediary results in nodeOutputs

            obj.nodeOutputs{obj.ANN.getNLayers()+1} = input;
            for layerN = obj.nLayers:-1:1
                input = obj.ANN.layers(1,layerN).getLayerOutput(input);
                obj.nodeOutputs{layerN} = input;
            end
            
            % fprintf('output: \t%f\n', input)
                        
        end % calculateNodeOutputs
        
        function calculateNodeErrors(obj, outputTarget)
            % Calculate the errors at a node given that the outputs have
            % already been calculated
            
            % calculate output layer error first
            networkOutput = obj.nodeOutputs{1};
            obj.nodeErrors{1} = networkOutput.*(1-networkOutput).*(outputTarget-networkOutput);
            
            % fprintf('error:  \tnetwork: %f, target: %d\n', networkOutput, outputTarget)
            
            % calculate hidden layer errors
            for layerN = 2:obj.nLayers
                % get all weights between this layer and next higher and
                % calculate this layer's node's error based off of the
                % layer above.
                
                % dot product of weights and nextError per node
                weights = obj.ANN.getLayerWeights(layerN-1); % matrix nOutputs x nNodes
                nextError = obj.nodeErrors{layerN-1}'; % nOutputs x 1
                nextNodeError = repmat(nextError, 1, obj.ANN.nHiddenNodes); % nOutputs x nNodes
                weightedErr = dot(weights, nextNodeError,1); % 1 x nNodes
                
                layerOutput = obj.nodeOutputs{layerN}; % 1 x nNodes
                obj.nodeErrors{layerN} = layerOutput.*(1-layerOutput).*weightedErr; % 1 x nNodes
            end
            
        end % calculateNodeErrors
        
        function updateWeights(obj)
            % once error rates have been calculated the node weights can be
            % updated
            
            for layerN = 1:obj.nLayers
                layer = obj.ANN.getLayer(layerN);
                
                for nodeN = 1:layer.getLayerSize()
                    weights = layer.nodes(nodeN).weights;
                    layerErr = obj.nodeErrors{layerN};
                    err = layerErr(nodeN);
                    inputs = obj.nodeOutputs{layerN + 1}; % output from layer below
                    % update weights
                    layer.nodes(nodeN).weights = weights + obj.learningRate*err*inputs;

                    delta = obj.learningRate*err*inputs;
                    % fprintf('weights:\t%f + %f = %f\n', weights, delta, layer.nodes(nodeN).weights) 
                    % fprintf('err:    \t%f\n', err)
                end
            end 
            
        end % updateWeights
        
    end % methods
    
end

