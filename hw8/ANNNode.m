classdef ANNNode
    % ANNNODE Node in an Artificial Neural Network (ANN).
    % Each node has weights and a threshold function
     
    properties
        nInputs;
        weights;
        thresholdFunct;
    end
    
    methods
        function obj = Node(nInputs, thresholdFunct)
            % constructor: init weights
            
            % TODO other ways to init wieght values
            obj.nInputs = nInputs;
            obj.weights = rand(1,nInputs) - .5; % range [ -.5, .5]
            
            obj.thresholdFunct = thresholdFunct;
        end % Node
        
        function o = getOutput(inputs)
            net = dot(inputs,obj.weights);
            o = obj.thresholdFunct(net);
        end % getOutput
        
        function updateWeights(newWeights)
            %TODO
            
        end % updateWeights
        
    end % methods
    
end % class

