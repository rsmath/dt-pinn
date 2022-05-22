function [ res ] = IsPointInObj( point,xeo,nr,h )
% Figures out if a point is inside an object

xe = xeo + 0.75*h*nr;
%xe = xeo;

%Find the nearest xe to "point"
[idx,~] = knnsearch(xe,point,'k',1);

loc_pt = xe(idx,:);
loc_nrml = nr(idx,:);

%Test against the normal to determine inside/outside
loc_vec = point-loc_pt;
dotP = dot(loc_vec,loc_nrml);

if dotP<1e-12
    res = 1;
else
    res = 0;
end

end

