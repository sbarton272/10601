classdef BackPropegation
    % BackPropegation Algorithm to train an ANN
    % Relies on the implementation of the ANN
    
    properties
        ANN;
        nIter = 0;
        learningRate = .01;
    end
    
    methods
        function obj = BackPropegation(learningRate)
            if (nargin > 0)
                obj.learningRate = learningRate;
                
            end
        end % BackPropegation
        
    end
    
end

