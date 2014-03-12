classdef ANNNode < handle
    % ANNNODE Node in an Artificial Neural Network (ANN).
    % Each node has weights and a threshold function
    % The node can calculate its output based on given inputs of the
    % correct size.
     
    properties
        nInputs;
        weights;
        thresholdFunct;
        weightFunct = @(len) (rand(1,len) - .5)/10; % range [-.05, .05]
    end
    
    methods
        function obj = ANNNode(nInputs, thresholdFunct)
            % constructor
            if (nargin > 0)
                % TODO other ways to init wieght values
                obj.nInputs = nInputs;
                % extra weight is offset weight
                obj.weights = obj.weightFunct(nInputs+1);

                obj.thresholdFunct = thresholdFunct;
            end
        end % Node
        
        function o = getOutput(obj, inputs)
            assert( isequal( size(inputs), [1, obj.nInputs] ), ...
              'ANNNode.getOutput: Inputs must be vector of correct size');
          % pad with a 1 for the offset weight
          inputs = [1, inputs];
          net = dot(inputs,obj.weights);
            o = obj.thresholdFunct(net);
        end % getOutput
                
        function setWeights(obj, newWeights)
            assert( isequal( size(newWeights), size(obj.weights) ), ...
                'ANNNode.updateWeights: newWeights must be vector of correct size');
            % weights includes offset weight
           obj.weights = newWeights;
        end
        
        function updateWeights(obj, newWeights)
            assert( isequal( size(newWeights), size(obj.weights) ), ...
              'ANNNode.updateWeights: newWeights must be vector of correct size');
            obj.weights = obj.weights + newWeights;
        end % updateWeights
                
    end % methods
    
end % class

