classdef sbfmodel2 < handle
    %SBFMODEL2 Forms a spherical basis function parametric model of a
    %closed curve in 2D.
    %Uses spherical polyharmonic splines to form parametric geometric
    %model given a set of Cartesian points in 2D. Assumes closed curve that
    %is simple (does not cross itself). Normals are also computed.
    
    properties        
        sbfexp
        sbf
        dsbfor
        s
        se
        A
        Ae       
        c
        xd
        xe
        tree
        ua
        la
        pa
        nre
        optsL
        optsU
        Nd
    end
    
    methods
        function obj = sbfmodel2(sbfexp)
            %SBFMODEL2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.sbfexp = sbfexp;
            obj.sbf = @(r) (r+eps).^obj.sbfexp;            
            obj.dsbfor = @(r) obj.sbfexp*(r+eps).^(obj.sbfexp-2);
            %obj.sbf = @(r) 1./sqrt(1 + (1.2*r).^2);
            %obj.dsbfor = @(r) -180./(36.*r.^2 + 25).^(3./2);
        end
        
        function obj = precompute(obj,Nd)
            %precompute Precomputes interpolation matrices and decomposes
            %them.
            %   Nd is the number of data sites.
            obj.s = linspace(-pi,pi,Nd+1); 
            obj.s = obj.s(1:end-1)';
            [si,sj] = ndgrid(obj.s,obj.s);    
            r = sqrt(2*(1 - cos(si-sj)));    
            obj.A = obj.sbf(r);  
            [obj.la,obj.ua,obj.pa] = lu(obj.A);
            obj.optsL.LT = true;
            obj.optsU.UT = true;
            obj.Nd = Nd;
        end
        
        function obj = buildModel(obj,xd)
            %buildModel Builds geometric model using precomputed
            %information
            % xd is the Nd x 2 Cartesian node location matrix
            if size(xd,1)~=obj.Nd
                error('xd is not the right size');
            end
            obj.xd = xd;
            obj.c = linsolve(obj.ua,linsolve(obj.la,obj.pa*obj.xd,obj.optsL),obj.optsU);
        end
        
        function obj = evaluateModel(obj,h)
            %evaluateModel Uses grid spacing h to evaluate geometric model
            % h is the desired Cartesian spacing of the sample sites
            
            %% Estimate the number of sample points we need using bounding box.
            [dims,~] = GetBoundingBoxSides(obj.xd);
            wd = dims(1); ht = dims(2);
            sa = 2*(wd+ht); %surface area
            Ne = floor(sa/h);

            %% Get super sampled boundary node set
            Ne_s = 2*Ne;
            se_s = linspace(-pi,pi,Ne_s+1); se_s = se_s(1:end-1)';
            [sie,sje] = ndgrid(se_s,obj.s);    
            re = sqrt(2*(1 - cos(sie-sje)));    
            obj.Ae = obj.sbf(re);    
            xe_s = obj.Ae*obj.c;
            De = obj.dsbfor(re);
            du = sin(sie - sje);
            De = du.*De;
            tr_s = -De*obj.c;
            nr_s = [-tr_s(:,2),tr_s(:,1)];
            nr_s = nr_s./repmat(sqrt(sum(nr_s.^2,2)),[1 2]);

            %% Thin the node set
            obj.tree = KDTreeSearcher(xe_s);
            flags = ones(length(xe_s),1);
            for k=1:Ne_s
                if(flags(k,1)~=0)
                    [idxs,~] = rangesearch(obj.tree,xe_s(k,:),0.9*h);
                    ids = idxs{:};       
                    flags(ids,1)=0;        
                    flags(k)=1;
                end
            end
            inds = flags==1;
            obj.se = se_s(inds,:);
            obj.xe = xe_s(inds,:);
            obj.nre = nr_s(inds,:);  
            obj.Ae = obj.Ae(inds,:);
        end        
    end
end

