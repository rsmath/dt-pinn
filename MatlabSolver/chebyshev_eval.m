function[p] = chebyshev_eval(x, alpha, d)
% chebyshev_eval -- Evaluates multidimensional Chebyshev polyhnoimals
%
% p = chebyshev_eval(x, alpha, {d = zeros([1 dim])} )
%
%   Evaluates Chebyshev polynomials that are orthonormal with respect to the
%   arcsine probability measure on [-1, 1]^dim. The dimension dim is equal to
%   both size(x,2), and size(alphas,2), and size(d, 2).
%
%   Each row of the matrix x is a point in dim-dimensional space, and each row
%   of the matrix alpha is a dim-dimensional multi-index. Each row of d is also
%   a multi-index specifying a partial derivative to be computed.
%
%     Input
%     -----------------
%     x         (M x dim) array, each row is a point on [-1,1]^d
%     alpha     (N x dim) array, each row is a dim-dimensional multi-index
%     d         (P x dim) array, each row is a dim-dimensional multi-index
%
%     Output
%     -----------------
%     p         (M x N x P) array, rows correspond to entries of x

[M,dim] = size(x);

if nargin < 3
  d = zeros([1 dim]);
else
  assert(size(d,2) == dim, 'Inputs x and d must have the same number of columns');
end

N = size(alpha,1);
P = size(d, 1);

assert(size(alpha,2)==dim, 'Inputs x and alpha must have the same number of columns');

p = ones([M N P]);

[a, b] = chebyshev_recurrence(max(alpha(:))+1);

for qd = 1:P
  
  for qdim = 1:dim
    temp = poly_eval(a, b, x(:,qdim), max(alpha(:,qdim)), d(qd,qdim) );

    p(:,:,qd) = p(:,:,qd).*temp(:,alpha(:,qdim)+1);
  end

end
