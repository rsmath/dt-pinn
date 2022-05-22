classdef domain 
    %DOMAIN A geometric domain on which a PDE is solved
    % This is a convenience class to hold node sets on geometric domains
    % where PDEs are solved. This class also contains methods for modifying
    % said geometric domain with embedded geometric objects, or for
    % classifying domain points as inside/outside the embedded objects.
    
    properties
        Xi_orig %all original inner points
        Xbo %outer boundary
        Xbi %inner boundary
        Xgo %outer ghost
        Xgi %inner ghost
        nri %inner normals
        nro %outer normals
        Xi %current set of inside points, accounting for refinement + modification by embedded object
        Xg %current full set of ghost points
        Xb %current full set of boundary points
        nr %current full set of boundary normals
        fcindex %index of first inner refinement point, these are "freshly-cleared"
        h %a measure of grid spacing
        Nobj %number of embedded objects
        point_type %inside or outside
        point_type_old %same, but at a previously time level, helps identify freshly-cleared
        X        
        Xf
        tree % a kd-tree
        int_tree %kd-tree on inner pts
        intbdry_tree
        Ni
        Nb
        Nbo
        Nbi
    end
    
    methods
        function obj = domain(Xi,Xb,nr,h)
            %DOMAIN Construct a domain using nodes and a spacing
            % Xi - initial set of interior points
            % Xb - outer boundary points
            % nr - outer boundary normals
            % h - spacing
            obj.Xi = Xi;
            obj.Xi_orig = Xi;
            obj.Xbo = Xb; obj.Xb = Xb;
            obj.Xgo = obj.Xbo + 0.25*h*nr;
            obj.Xg = obj.Xgo;
            obj.nro = nr; obj.nr = nr;
            obj.X = [obj.Xi;obj.Xb];
            obj.Xf = [obj.X;obj.Xg];
            obj.h = h;
            obj.tree = KDTreeSearcher(obj.Xf);
            obj.intbdry_tree = KDTreeSearcher(obj.X);
            obj.int_tree = KDTreeSearcher(obj.Xi);
            obj.Ni = size(obj.Xi,1);
            obj.Nb = size(obj.Xb,1);
            obj.Nbo = obj.Nb;
            obj.point_type = zeros(size(obj.Xi_orig,1),1);
            %obj.point_type_old = zeros(size(obj.Xi_orig,1),1);
        end
        
        function obj = adjustDomain(obj,geom_models,f)
            %adjustDomain Takes a set of embedded objects represented by
            %SBF models, adjusts the domain to turn off any points that are
            %inside the embedded objects 
            
            %% Get inner boundary points
            obj = obj.getInnerBdryPoints(geom_models);
            
            
            %% first classify as inside/outside
            obj.point_type = ClassifyPoints(obj.Xi_orig,geom_models,obj.h);
            
           
            Xi_inner_refined = obj.Xbi + 0.5*obj.h*obj.nri; %near-boundary refinement
            obj.Xi = obj.Xi_orig(obj.point_type(:,1)==0,:);
            %obj.fcindex = length(obj.Xi)+1; %index of first inner refinement point, also implicitly a freshly-cleared point
            obj.Xi = [obj.Xi;Xi_inner_refined];
            obj.Xb = [obj.Xbo;obj.Xbi];
            obj.X = [obj.Xi;obj.Xb];
            obj.Xg = [obj.Xgo;obj.Xgi];
            obj.Xf = [obj.X;obj.Xg];
            obj.nr = [obj.nro;-obj.nri]; 
            obj.tree = KDTreeSearcher(obj.Xf);
            obj.Ni = size(obj.Xi,1);
            obj.Nb = size(obj.Xb,1);
            obj.Nbo = size(obj.Xbo,1);
            obj.Nbi = size(obj.Xbi,1);
        end
        
        function c_Omnp1 = transferFieldBSL(obj,Xe,c_Omn,interp_struct_n,rbf)
            %transferSolution Transfers field cn on THIS domain to new
            %domain and return. Only interior and boundary points.          
            % c_Omn - field defined on this domain            
            % Omn - old domain
            % interp_struct_n - interpolation struct on Omn
            % Omnp1 - new domain
            % c_Omnp1 - transferred field

            %% For transfer, we'll need a tree of stencil center.
            stencil_tree = interp_struct_n{end,1};
            recurrence = interp_struct_n{end,2};
            a = interp_struct_n{end,3};
            
            
            %% Get nearest stencil info and sort
            [idx,~] = knnsearch(stencil_tree,Xe,'k',1);
            
            %% Do transfer
            %c_Omnp1 = TransferSolution(idx,c_Omn,Xe,interp_struct_n,rbf,a,recurrence);            
            c_Omnp1 = TransferSolutionBatched(idx,c_Omn,Xe,interp_struct_n,rbf,a,recurrence);            
        end  
        
        function c_Omnp1 = transferField(obj,c_Omn,interp_struct_n,rbf)
            %transferSolution Transfers field cn on THIS domain to new
            %domain and return. Only interior and boundary points.          
            % c_Omn - field defined on this domain            
            % Omn - old domain
            % interp_struct_n - interpolation struct on Omn
            % Omnp1 - new domain
            % c_Omnp1 - transferred field

            %% For transfer, we'll need a tree of stencil center.
            X_stencil = cell2mat(interp_struct_n(:,1));
            stencil_tree = KDTreeSearcher(X_stencil);                   
            
            %% First transfer data to the interior and boundary points
            [idx,~] = knnsearch(stencil_tree,obj.X,'k',1);
            c_Omnp1 = TransferSolution(idx,c_Omn,obj.X,interp_struct_n,rbf);

        end        
        function cg = fillGhostCells(obj, c,g,Dn)
            Bbg = Dn(:,(obj.Ni + obj.Nb)+1:end);
            Bbi = Dn(:,1:obj.Ni);
            Bbb = Dn(:,obj.Ni+1:obj.Ni+obj.Nb);
            ci = c(1:obj.Ni,:); cb = c(obj.Ni+1:end,:);
            cg = Bbg\(g - (Bbi*ci + Bbb*cb));            
        end
    end
    methods (Access = private)
        function obj = getInnerBdryPoints(obj,geom_models)
            %computeInnerBdryPoints Takes a set of embedded objects
            %represented by SBF models, grabs their sample sites  
            obj.Xbi = [];
            obj.nri = [];
            obj.Nobj = size(geom_models,1);
            for k=1:obj.Nobj
                obj.Xbi = [obj.Xbi;geom_models(k,1).xe];
                obj.nri = [obj.nri;geom_models(k,1).nre];
            end
            obj.Xgi = obj.Xbi - 0.5*obj.h*obj.nri;            
            obj.Nb = size(obj.Xb,1);            
        end
    end
end

