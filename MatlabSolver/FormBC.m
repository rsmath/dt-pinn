function [Dn,Omnp1_bdry_is] = FormBC(Neucoeff,Dircoeff,Xb,nr,rbf,drbf,tree,stencilSize,ell)
%%Non-overlapped approximation to boundary operators

s_dim = size(Xb,2);
Np = length(Xb);
tX = tree.X;
row_index = zeros(stencilSize,Np);
col_index = row_index;
wghts_dn = row_index;

tarray = ones(1,stencilSize);


recurrence = @(N) jacobi_recurrence(N,0,0);
a = total_degree_indices(s_dim,ell);
npoly = size(a,1);
Omnp1_bdry_is = cell(Np,12);


%figure
for iter=1:Np
    
    if Neucoeff(iter)==0 && Dircoeff(iter)~=0
        dit = Dircoeff(iter)*speye(1,stencilSize);
        [je,~] = knnsearch(tree,Xb(iter,:),'k',stencilSize);
        p = tX(je,1:s_dim);    
        Omnp1_bdry_is{iter,1} = Xb(iter,:);
        Omnp1_bdry_is{iter,2} = je;
        Omnp1_bdry_is{iter,3} = p;        
        Omnp1_bdry_is{iter,4} = [];
        Omnp1_bdry_is{iter,5} = [];
        Omnp1_bdry_is{iter,6} = [];
        Omnp1_bdry_is{iter,7} = npoly;    
        Omnp1_bdry_is{iter,8} = ell;    
        Omnp1_bdry_is{iter,9} = [];
        Omnp1_bdry_is{iter,10} = [];    
        Omnp1_bdry_is{iter,11} = iter;
        Omnp1_bdry_is{iter,12} = tX(iter,:);         
    else
        [je,~] = knnsearch(tree,Xb(iter,:),'k',stencilSize);
        p = tX(je,1:s_dim);    
        rd = sqrt(max(bsxfun(@plus,sum(p.*p,2),sum(p.*p,2)') - 2*(p*p'),0));           
        ns = nr(iter,:);

        %% Local augmented RBF interpolation matrix;
        Ar = rbf(1,rd);        
        w2 = rd(1,stencilSize);
        cm = mean(p);
        %pc = (p - repmat(cm,[stencilSize,1]))/w2;    
        v = mpoly_eval(p,a,recurrence);      
        A = [[Ar v];[v.' zeros(npoly,npoly)]];

        %% Decompose the matrix in LUP style
        [la,ua,pa] = lu(A); 

        if s_dim==2
            dva = mpoly_eval(p(1,:),a,recurrence,[1,0]);
            dvb = mpoly_eval(p(1,:),a,recurrence,[0,1]);
            %gpx = gpx./w2;
            %gpy = gpy./w2;
        elseif s_dim==3
            dva = mpoly_eval(p(1,:),a,recurrence,[1,0,0]);
            dvb = mpoly_eval(p(1,:),a,recurrence,[0,1,0]);
            dvc = mpoly_eval(p(1,:),a,recurrence,[0,0,1]);
            %gpx = gpx./w2;
            %gpy = gpy./w2;        
            %gpz = gpz./w2;
        end

        p1 = zeros(1,npoly);        
        p1(1,:) = ns(1).*dva(1,:) + ns(2).*dvb(1,:);
        if s_dim==3
            p1(1,:) = p1(1,:)+ ns(3).*dvc(1,:);
        end

        ve = mpoly_eval(p(1,:),a,recurrence);
        gp = Neucoeff(iter)*p1 + Dircoeff(iter)*ve;

        [xj,xk] = ndgrid(p(1,1),p(:,1));
        [yj,yk] = ndgrid(p(1,2),p(:,2));    
        xdiff = xj-xk;
        ydiff = yj-yk;    
        if s_dim==3
            [zj,zk] = ndgrid(p(1,3),p(:,3));    
            zdiff = zj-zk;
        end
        D = drbf(1,rd(1,:));

        p2 = (ns(1)*(xdiff) + ns(2)*(ydiff)).*D;
        if s_dim==3
            p2 = p2 + ns(3)*(zdiff).*D;
        end
        B = [(Neucoeff(iter)*p2 + Dircoeff(iter)*Ar(1,:)),gp];
        %B = [p2,p1];

        %% Compute all weights and throw out a few.    
        dit = ((B/ua)/la)*pa;
       % dit = Neucoeff(iter)*dit + Dircoeff(iter);
        
        Omnp1_bdry_is{iter,1} = Xb(iter,:);
        Omnp1_bdry_is{iter,2} = je;
        Omnp1_bdry_is{iter,3} = p;        
        Omnp1_bdry_is{iter,4} = la;
        Omnp1_bdry_is{iter,5} = ua;
        Omnp1_bdry_is{iter,6} = pa;
        Omnp1_bdry_is{iter,7} = npoly;    
        Omnp1_bdry_is{iter,8} = ell;    
        Omnp1_bdry_is{iter,9} = cm;
        Omnp1_bdry_is{iter,10} = w2;    
        Omnp1_bdry_is{iter,11} = iter;
        Omnp1_bdry_is{iter,12} = Xb(iter,:);        
    end
    %% Compute norms of center weights    
    wghts_dn(:,iter) = dit(1:stencilSize);
    row_index(:,iter) = iter.*tarray;
    col_index(:,iter) = je; 
end
%% Build diff mat
Dn = sparse(row_index(:),col_index(:),wghts_dn(:),Np,size(tree.X,1));

X_stencil = cell2mat(Omnp1_bdry_is(:,1));
stencil_tree = KDTreeSearcher(X_stencil);    
Omnp1_bdry_is{iter+1,1} = stencil_tree;
Omnp1_bdry_is{iter+1,2} = recurrence;
Omnp1_bdry_is{iter+1,3} = a;
end