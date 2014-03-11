classdef ANNNodeLayer < handle
    % ANNNodeLayer Layer of nodes
    % Keeps track of layer size and calculates output
    
    properties
        layerSize;
        nodes;
    end
    
    methods
        function obj = ANNNodeLayer(layerSize, nNodeInputs, nodeThreshFunct)
            if (nargin > 0)
                obj.layerSize = layerSize;
                
                % prealloc first
                layerNodes(1, obj.layerSize) = ANNNode();
                for n = 1:obj.layerSize
                    layerNodes(1,n) = ANNNode(nNodeInputs, nodeThreshFunct);
                end
                obj.nodes = layerNodes; % better performance to assign later
                
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
        
        function weights = getWeights(obj)
            for nodeN = obj.layerSize:-1:1
                weights(nodeN,:) = obj.nodes(nodeN).weights;
            end
        end % getWeights
        
        function layerSize = getLayerSize(obj)
            layerSize = obj.layerSize;
        end    
        
    end
    
end

