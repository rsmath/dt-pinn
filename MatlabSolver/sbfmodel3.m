classdef sbfmodel3 < handle
    %SBFMODEL3 Forms a spherical basis function parametric model of a
    %closed surface in 3D.
    %Uses spherical polyharmonic splines to form parametric geometric
    %model given a set of Cartesian points in 3D. Assumes closed surface that
    %is simple (does not cross itself). Normals are also computed.
    
    properties        
        %% rbf properties
        sbfexp
        sbf
        dsbfor
        
        %% parameteric vars at datasites and sample sites
        u
        v
        ue
        ve
        
        %% Interpolation and diff mats at data sites
        A
        Du
        Dv
        
        %% Sample site eval matrix
        Ae       
        
        %% Interpolation coefficients
        c
        
        %% data sites, samples sites and a kd-tree for elimination
        xd
        xe
        tree

        %% Normals at data sites and sample sites
        nr
        nre
        
        %% LU factors of A
        optsL
        optsU
        ua
        la
        pa       
        
        %% sizes
        Nd
        Ne
        
        %% QR factors of Ae
        Q
        R
        p
    end
    
    methods
        function obj = sbfmodel3(sbfexp)
            %SBFMODEL2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.sbfexp = sbfexp;
            obj.sbf = @(r) (r).^obj.sbfexp.*log(r+eps);            
            obj.dsbfor = @(r) r.^(obj.sbfexp - 2).*(obj.sbfexp*log(r+eps) + 1);
            %obj.sbf = @(r) 1./sqrt(1 + (1.2*r).^2);
            %obj.dsbfor = @(r) -180./(36.*r.^2 + 25).^(3./2);
        end
        
        function obj = precompute(obj,Nd)
            %precompute Precomputes interpolation matrices and decomposes
            %them.
            %   Nd is the number of data sites.
            x = spiral_points(1,[0,0,0],Nd);
            [obj.u,obj.v] = cart2sphm(x);
            [ui,uj] = ndgrid(obj.u,obj.u);
            [vi,vj] = ndgrid(obj.v,obj.v);
            r = 2*(1 - cos(vi).*cos(vj).*cos(ui-uj) - sin(vi).*sin(vj));
            r = sqrt(abs(r));
            obj.A = obj.sbf(r);
            du = cos(vi).*cos(vj).*sin(ui-uj);
            dv = sin(vi).*cos(vj).*cos(ui-uj) - cos(vi).*sin(vj);
            Dl = obj.dsbfor(r);
            obj.Du = Dl.*du;
            obj.Dv = Dl.*dv;            
            
            [obj.la,obj.ua,obj.pa] = lu(obj.A);
            obj.optsL.LT = true;
            obj.optsU.UT = true;
            obj.Nd = Nd;
        end
        
        function obj = buildModel(obj,xd)
            %buildModel Builds geometric model using precomputed
            %information
            % xd is the Nd x 3 Cartesian node location matrix
            if size(xd,1)~=obj.Nd
                error('xd is not the right size');
            end
            obj.xd = xd;
            obj.c = linsolve(obj.ua,linsolve(obj.la,obj.pa*obj.xd,obj.optsL),obj.optsU);
        end
        
        function obj = evaluateModel(obj,h)
            %evaluateModel Uses grid spacing h to evaluate geometric model
            % h is the desired Cartesian spacing of the sample sites
            
            %% First evaluate everything at data sites
            t1 = obj.Du*obj.c;
            t2 = obj.Dv*obj.c;
            obj.nr = cross(t1,t2,2); 
            obj.nr = (obj.nr)./repmat(sqrt(sum((obj.nr).^2,2)),[1 3]);

            
            %% Now evaluate at sample sites with supersampling and thinning
            %% Estimate the number of sample points we need using bounding box.
            [dims,~] = GetBoundingBoxSides(obj.xd);
            w1 = dims(1); w2 = dims(2); w3 = dims(3);
            sa = 2*(w1*+w2 + w2*w3 + w1*w3); %surface area
            Ne = floor(sa/h^2);

            %% Get super sampled boundary node set
            Ne_s = 5*Ne;
            lxe = spiral_points(1,[0,0,0],Ne_s);
            [ue_s,ve_s] = cart2sphm(lxe);    
            [uie,uje] = ndgrid(ue_s,obj.u);
            [vie,vje] = ndgrid(ve_s,obj.v);
            re = 2*(1 - cos(vie).*cos(vje).*cos(uie-uje) - sin(vie).*sin(vje));
            re = sqrt(abs(re));
            obj.Ae = obj.sbf(re);   
            xe_s = obj.Ae*obj.c;
            du = cos(vie).*cos(vje).*sin(uie-uje);
            dv = sin(vie).*cos(vje).*cos(uie-uje) - cos(vie).*sin(vje);
            Dl = obj.dsbfor(re);
            D1u = Dl.*du;
            D1v = Dl.*dv;
            t1e_s = D1u*obj.c;
            t2e_s = D1v*obj.c;
            nr_s = cross(t1e_s,t2e_s,2); 
            nr_s = nr_s./repmat(sqrt(sum(nr_s.^2,2)),[1 3]);

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
            obj.ue = ue_s(inds,:);
            obj.ve = ve_s(inds,:);
            obj.xe = xe_s(inds,:);
            obj.nre = nr_s(inds,:);  
            obj.Ae = obj.Ae(inds,:);
            obj.Ne = length(obj.ue);
            %[obj.Q,obj.R,obj.p] = qr(obj.Ae,0);
        end  
        
        function ceval = Expand(obj,c)
            %% Takes a quantity c defined at xd and evalutes at xe
            coeff = linsolve(obj.ua,linsolve(obj.la,obj.pa*c,obj.optsL),obj.optsU);
            ceval = obj.Ae*coeff;
        end
        
        function cd = Contract(obj,ceval)
            %% Takes a quantity c defined at xe and evalutes at xd
            coeff = zeros(obj.Nd,size(ceval,2));
            coeff(obj.p,:) = obj.R\(obj.Q\ceval);
            cd = obj.A*coeff;
        end        
    end
end

