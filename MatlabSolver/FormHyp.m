function [H_rbf,cnta] = FormHyp(Xd,rbfpow,tree,sz,ell,hyppow)
%%Instead of throwing away the weights computed by each RBF, we retain the
%%weights for nodes "delta" out to the border of each stencil. The result
%%is less work in total. delta is a number between 0 and 1. delta = 1
%%recovers the standard RBF-FD method, delta = 0 is the other extreme.


Np = length(Xd);
row_index = zeros(sz,Np);
col_index = row_index;
wghts_hyp = row_index;

ind_list = 1:Np; %Maintain list of global indices
iter = 1;
cnta = 1;
tarray = ones(1,sz);
s_dim = size(Xd,2);

alph = 0; bet = 0;
recurrence = @(N) jacobi_recurrence(N,alph,bet);
a = total_degree_indices(s_dim,ell);
npoly = size(a,1);


while iter <= ind_list(end)
  
    [je,~] = knnsearch(tree,Xd(iter,:),'k',sz);
    p = tree.X(je,1:s_dim);
    nd = sz;
    rd = sqrt(max(bsxfun(@plus,sum(p.*p,2),sum(p.*p,2)') - 2*(p*p'),0));               

   
    %% Local augmented RBF interpolation matrix
    A = lrbf(rd,0,s_dim,rbfpow);    
    w2 = rd(1,nd);
    cm = p(1,:);
    pc = (p - repmat(cm,[nd,1]))/w2;    
    v = mpoly_eval(pc,a,recurrence);
    hv = zeros(size(v));
    A = [[A v];[v.' zeros(npoly,npoly)]];   
    
    %% Decompose the matrix in LUP style    
    [la,ua,pa] = lu(A); 
    
    %% Compute first weight and norms
    rdit = rd(1,:);
    Br = lrbf(rdit,hyppow,s_dim,rbfpow);
    B = [Br,hv(1,:)];
    hit = ((B/ua)/la)*pa;
    nh1 = abs(hit*A*hit');
    nl1 = norm(hit,1);
    wghts_hyp(:,je(1)) = hit(1:nd);
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
        Br = lrbf(rdit,hyppow,s_dim,rbfpow);
        B = [Br,hv(it,:)];    
    
        %% Compute weights
        hit = ((B/ua)/la)*pa;        
        
        %% Compute norm of weights
        nh = abs(hit*A*hit');
        nl = norm(hit,1);
        
        %% compare against norm of stencil center and break loop if larger
        if nh > nh1 || nl>nl1
           continue;
        end 
        it_list(it_cnt)= it;
    
        t1 = je(it);
        g = ismember(t1,ind_list);
        if g~=0
            wghts_hyp(:,t1) = hit(1:nd);
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
H_rbf = sparse(row_index(:),col_index(:),wghts_hyp(:),Np,size(tree.X,1));


end