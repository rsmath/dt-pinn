function [ res ] = lrbf( r, k ,d, m )
%LRBF Powers k of laplacian of PHS rbf of order m in d dimensions

cres = 1;
for it = 1:k        
    cres = cres*m*(d + m - 2.0);  
    m = m-2;      
end
res = cres*r.^m;    
end

