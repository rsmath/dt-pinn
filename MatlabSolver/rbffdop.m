classdef rbffdop
    %RBFFDOP Properties of a generic diff. op. approximated by rbf-fd
    %  This class provides a way to set approximation properties for 
    % RBF-FD approximations to differential operators.
    
    properties
        ell
        polyM
        stencilSize         
        rbfexp
        rbf
        drbfor
        d2rbf
        s_dim
        theta
        xi
        hyppow %only used for hyperviscosity operator
    end
    
    methods
        function obj = rbffdop(s_dim, xi, theta,flag)
            %RBFFDOP Construct an instance of this class
            %   Set RBF-FD approximation properties.
            % s_dim_in: spatial dimension
            % xi_in: desired order of spatial approximation
            % theta_in: order of differential operator
            
            if flag==0
                
                
                obj.xi = xi;
                obj.theta = theta;
                obj.s_dim = s_dim;
 %               if theta~=0
                    obj.ell = obj.xi + obj.theta - 1;
                    if mod(obj.ell,2)==0
                        obj.rbfexp = obj.ell-1;
                    else
                        obj.rbfexp = obj.ell;
                    end
                    obj.rbfexp = max([obj.rbfexp,5]);        
                    obj.rbfexp = min([obj.rbfexp,11]);          
%                 else                    
%                     obj.ell = xi;
%                     obj.rbfexp = 2*xi+1;
%                 end
                obj.rbf = @(ep,r) (r+eps).^obj.rbfexp;
                obj.drbfor = @(ep,r) obj.rbfexp*(r+eps).^(obj.rbfexp-2);
                obj.d2rbf = @(ep,r) obj.rbfexp.*(obj.rbfexp-1).*(r+eps).^(obj.rbfexp-2);        
                obj.polyM = nchoosek(obj.ell+obj.s_dim,obj.s_dim);
                obj.stencilSize = 2*obj.polyM+1;
                
            else %hyperviscosity
                obj.s_dim = s_dim;
                
                %% use laplacian as judge of hyppow
                obj.xi = 1; 
                obj.ell = obj.xi + 1;
                obj.polyM = nchoosek(obj.ell+obj.s_dim,obj.s_dim);
                obj.stencilSize = 2*obj.polyM+1;   
                obj.hyppow = floor(1.5*log(obj.stencilSize)); 
                obj.rbfexp = 2*obj.hyppow+1;
                obj.ell = obj.hyppow;
                obj.polyM = nchoosek(obj.ell+obj.s_dim,obj.s_dim);
                obj.stencilSize = 2*obj.polyM + 1;
            end
        end
    end
end

