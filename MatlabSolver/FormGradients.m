function [Gx_rbf,Gy_rbf,Gz_rbf,cnta,interp_struct] = FormGradients(Xd,rbf,drbf,tree,sz,ell)
%%Attempts to compute differential operators without an overlap parameter.
%The approach is to first compute a weight at the center, then
%compute its local L-Lebesgue function. For each subsequent point in
%stencil, compute the same, compare against center, and one
%ends up being worse than the center, reject that one and move onto the
%next one.

s_dim = size(Xd,2);
Np = length(Xd);
row_index = zeros(sz,Np);
col_index = row_index;
wghts_gx = row_index;
wghts_gy = row_index;
if s_dim==3
    wghts_gz = row_index;
end

ind_list = 1:Np; %Maintain list of global indices
iter = 1;
cnta = 1;
tarray = ones(1,sz);


alph = 0; bet = 0;
recurrence = @(N) jacobi_recurrence(N,alph,bet);
a = total_degree_indices(s_dim,ell);
npoly = size(a,1);

estimated_num_stencils = floor(0.25*Np);
interp_struct = cell(estimated_num_stencils,10);

%figure
while iter <= ind_list(end)  
    
    [je,~] = knnsearch(tree,Xd(iter,:),'k',sz);
    p = tree.X(je,1:s_dim);
    nd = sz;   
    [xk,xj] = meshgrid(p(:,1));
    [yk,yj] = meshgrid(p(:,2));        
    if s_dim==3
        [zk,zj] = meshgrid(p(:,3));            
    end
    rd = sqrt(max(bsxfun(@plus,sum(p.*p,2),sum(p.*p,2)') - 2*(p*p'),0));           
    
    interp_struct{cnta,1} = Xd(iter,:);
    interp_struct{cnta,2} = je;
    interp_struct{cnta,3} = p;        
    
    %% Local augmented RBF interpolation matrix;
    A = rbf(1,rd);        
    w2 = rd(1,nd);
    cm = p(1,:);
    pc = (p - repmat(cm,[nd,1]))/w2;     
    v = mpoly_eval(pc,a,recurrence);  
    if s_dim==2
        gpx = mpoly_eval(pc,a,recurrence,[1,0]);
        gpy = mpoly_eval(pc,a,recurrence,[0,1]);
        gpx = gpx./w2;
        gpy = gpy./w2;
    elseif s_dim==3
        gpx = mpoly_eval(pc,a,recurrence,[1,0,0]);
        gpy = mpoly_eval(pc,a,recurrence,[0,1,0]);
        gpz = mpoly_eval(pc,a,recurrence,[0,0,1]);
        gpx = gpx./w2;
        gpy = gpy./w2;        
        gpz = gpz./w2;
    end
  
    A = [[A v];[v.' zeros(npoly,npoly)]];
    
    %% Decompose the matrix in LUP style
    [la,ua,pa] = lu(A); 
    
    interp_struct{cnta,4} = la;
    interp_struct{cnta,5} = ua;
    interp_struct{cnta,6} = pa;
    interp_struct{cnta,7} = npoly;    
    interp_struct{cnta,8} = ell;    
    interp_struct{cnta,9} = cm;
    interp_struct{cnta,10} = w2;
    
    %% Compute first weight and norms
    rdit = rd(1,:);
    D = drbf(1,rdit);         
    
    Bx = [(xj(1,:) - xk(1,:)).*D,gpx(1,:)];
    By = [(yj(1,:) - yk(1,:)).*D,gpy(1,:)];
    gxit = ((Bx/ua)/la)*pa;
    gyit = ((By/ua)/la)*pa;    
    lebnormgx1 = norm(gxit,1); lebnormgy1 = norm(gyit,1);
    nsnormgx1 = abs(gxit*A*gxit'); nsnormgy1 = abs(gyit*A*gyit');  
    wghts_gx(:,je(1)) = gxit(1:nd);
    wghts_gy(:,je(1)) = gyit(1:nd);
    if s_dim==3        
        Bz = [(zj(1,:) - zk(1,:)).*D,gpz(1,:)];
        gzit = ((Bz/ua)/la)*pa;        
        lebnormgz1 = norm(gzit,1); 
        nsnormgz1 = abs(gzit*A*gzit');
        wghts_gz(:,je(1)) = gzit(1:nd);
    end
    
    row_index(:,je(1)) = je(1).*tarray;
    col_index(:,je(1)) = je;    
    
    %% Compute as many weights as we safely can for this stencil
    %%Keep track of which ones we computed weights for
    it_list = zeros(1,nd);
    it_list(1) = 1;
    it_cnt = 2;
    for it=2:nd
        %% Compute evaluation operators
        rdit = rd(it,:);
        D = drbf(1,rdit); 
        Bx = [(xj(it,:) - xk(it,:)).*D,gpx(it,:)];
        By = [(yj(it,:) - yk(it,:)).*D,gpy(it,:)];
        if s_dim==3
             Bz = [(zj(it,:) - zk(it,:)).*D,gpz(it,:)];
        end
    
        %% Compute weights
        gxit = ((Bx/ua)/la)*pa;
        gyit = ((By/ua)/la)*pa;
        if s_dim==3
           gzit = ((Bz/ua)/la)*pa; 
        end
        
        %% Compute norm of weights
        lebnormgx = norm(gxit,1); lebnormgy = norm(gyit,1);
        nsnormgx = abs(gxit*A*gxit'); nsnormgy = abs(gyit*A*gyit');
        if s_dim==3
           lebnormgz = norm(gzit,1); nsnormgz = abs(gzit*A*gzit');
        end
        
        %% compare against norm of stencil center and break loop if larger
        if s_dim==2
            if lebnormgx > lebnormgx1 || lebnormgy > lebnormgy1 || nsnormgx> nsnormgx1 || nsnormgy>nsnormgy1
                continue;
            end 
        elseif s_dim==3
            if lebnormgx > lebnormgx1 || lebnormgy > lebnormgy1 || lebnormgz > lebnormgz1 || nsnormgx> nsnormgx1 || nsnormgy>nsnormgy1 || nsnormgz>nsnormgz1 ||lebnormgz>lebnormgz1
                continue;
            end 
        end
        it_list(it_cnt)= it;
    
        t1 = je(it);
        g = ismember(t1,ind_list);
        if g~=0            
            wghts_gx(:,t1) = gxit(1:nd);
            wghts_gy(:,t1) = gyit(1:nd);
            if s_dim==3
                wghts_gz(:,t1) = gzit(1:nd);
            end
            row_index(:,t1) = t1.*tarray;
            col_index(:,t1) = je; 
        else
            it_list(it_cnt) = 0;
        end
        it_cnt = it_cnt+1;
    end
    it_list = nonzeros(it_list);
    
    %% Modify the index list to not include the indices for the nodes we've
    % already computed the weights.
    t1 = sort(je(it_list));
    for ii=1:length(t1)
        ind_list(find(ind_list == t1(ii),1,'first'))= [];
    end
    
    if isempty(ind_list)
        break;
    else
        iter = ind_list(1);
        cnta = cnta+1;
    end
end
%% Build diff mats
Gx_rbf = sparse(row_index(:),col_index(:),wghts_gx(:),Np,size(tree.X,1));
Gy_rbf = sparse(row_index(:),col_index(:),wghts_gy(:),Np,size(tree.X,1));
if s_dim==3
    Gz_rbf = sparse(row_index(:),col_index(:),wghts_gz(:),Np,size(tree.X,1));
else
    Gz_rbf = [];
end

end