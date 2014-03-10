classdef ANNNodeLayer < handle
    % ANNNodeLayer Layer of nodes
    % Keeps track of layer size and calculates output
    
    properties
        layerSize;
        nodes = [];
    end
    
    methods
        function obj = ANNNodeLayer(layerSize, nNodeInputs, nodeThreshFunct)
            if (nargin > 0)
                obj.layerSize = layerSize;
                
                % init backwards important for pre allocation
                for n = obj.layerSize:-1:1
                    nodes(1,n) = ANNNode(nNodeInputs, nodeThreshFunct);
                end
                obj.nodes = nodes; % better performance to assign later
                
            end
        end % ANNNodeLayer
        
        function output = getLayerOutput(obj,input)
       
            % prealloc
            output = 1:obj.layerSize;
            % loop is more efficient then arrayfun
            for n = 1:obj.layerSize
                output(n) = obj.nodes(n).getOutput(input);
            end
            
        end % getLayerOutput
        
    end
    
end

