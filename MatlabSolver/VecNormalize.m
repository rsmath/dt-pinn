function [ xn ] = VecNormalize( x )
%This function safely normalizes a list of vectors.
%Input: x- an array of vectors, each row representing an nxd vector
%Output: xn - an array of unit vectors corresponding to x

d = sqrt(sum(x.^2,2));
d(d<1e-10) = 1;
xn=x./repmat(d,[1 3]);

end

