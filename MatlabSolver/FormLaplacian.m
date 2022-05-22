function [L_rbf,cnta,interp_struct] = FormLaplacian(Xd,rbf,drbf,d2rbf,tree,sz,ell)
%%Compute differential operators without an overlap parameter.
%The approach is to first compute a weight at the center, then
%compute its local L-Lebesgue function. For each subsequent point in
%stencil, compute the same, compare against center, and one
%ends up being worse than the center, reject that one and move onto the
%next one. We'll also check native space norms to ensure the weights are
%not too oscillatory.


Np = length(Xd);
row_index = zeros(sz,Np);
col_index = row_index;
wghts_lap = row_index;

%eps = zeros(Np,1);

ind_list = 1:Np; %Maintain list of global indices
iter = 1;
cnta = 1;
tarray = ones(1,sz);
s_dim = size(Xd,2);

recurrence = @(N) jacobi_recurrence(N,0,0);
a = total_degree_indices(s_dim,ell);
npoly = size(a,1);

estimated_num_stencils = floor(0.05*Np);
interp_struct = cell(estimated_num_stencils,12);
tX = tree.X;
%figure
while iter <= ind_list(end)  
    
    [je,~] = knnsearch(tree,Xd(iter,:),'k',sz);    
    je = je';
    p = tX(je,1:s_dim);    
    nd = sz;
    rd = sqrt(max(bsxfun(@plus,sum(p.*p,2),sum(p.*p,2)') - 2*(p*p'),0));      
    
    interp_struct{cnta,1} = Xd(iter,:);
    interp_struct{cnta,2} = je;
    interp_struct{cnta,3} = p;    
        
    
    %% Local augmented RBF interpolation matrix;
    Ar = rbf(1,rd);   
    w2 = rd(1,nd);
    cm = mean(p);
    %pc = (p - repmat(cm,[nd,1]))/w2;    
    v = mpoly_eval(p,a,recurrence);
    if s_dim==2
        lp = mpoly_eval(p,a,recurrence,[2,0]) + mpoly_eval(p,a,recurrence,[0,2]);        
    elseif s_dim==3
        lp = mpoly_eval(p,a,recurrence,[2,0,0])...
             + mpoly_eval(p,a,recurrence,[0,2,0])...
             + mpoly_eval(p,a,recurrence,[0,0,2]);            
    end    
    A = [[Ar v];[v.' zeros(npoly,npoly)]];
    
    %% Decompose the matrix in LUP style
    [la,ua,pa] = lu(A); 
    
    interp_struct{cnta,4} = la;
    interp_struct{cnta,5} = ua;
    interp_struct{cnta,6} = pa;
    interp_struct{cnta,7} = npoly;    
    interp_struct{cnta,8} = ell;    
    interp_struct{cnta,9} = cm;
    interp_struct{cnta,10} = w2;
    
    %% Compute all rhs vals at once for BLAS3 efficiency.
    D = drbf(1,rd);
    B = [d2rbf(1,rd) + (s_dim-1)*D,lp];
    
    %% Compute all weights and throw out a few.    
    full_weights = ((B/ua)/la)*pa;
    
    %% Compute norms of center weights    
    lit = full_weights(1,:);
    nsnorm1 = abs(lit*A*lit');    
    lebnorm1 = norm(lit,1);    
    wghts_lap(:,je(1)) = lit(1:nd);    
    row_index(:,je(1)) = je(1).*tarray;
    col_index(:,je(1)) = je;    

    %% Keep as many weights as we safely can for this stencil
    %%Keep track of which ones we computed weights for
    it_list = zeros(1,nd);
    it_list(1) = 1;
    it_cnt = 2;
    for it=2:nd
        %% Compute weights
        lit = full_weights(it,:);
        
        %% Compute norm of weights        
        lebnorm = norm(lit,1);
        nsnorm = abs(lit*A*lit');
        
        %% compare against norm of stencil center and skip loop iteration
        if nsnorm > nsnorm1 || lebnorm> lebnorm1
           continue;
        end 
        it_list(it_cnt)= it;
    
        t1 = je(it);
        g = ismember(t1,ind_list);
        if g~=0
            wghts_lap(:,t1) = lit(1:nd);
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
    interp_struct{cnta,11} = t1;
    interp_struct{cnta,12} = tX(t1,:);
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
%% Build diff mat
L_rbf = sparse(row_index(:),col_index(:),wghts_lap(:),Np,length(tX));

%% build kd-tree on stencil centers and pack back into interp
X_stencil = cell2mat(interp_struct(:,1));
stencil_tree = KDTreeSearcher(X_stencil);    
interp_struct{cnta+1,1} = stencil_tree;
interp_struct{cnta+1,2} = recurrence;
interp_struct{cnta+1,3} = a;
end