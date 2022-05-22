function[p] = mpoly_eval(x, alpha, recurrence, d)
% mpoly_eval -- Evaluates tensorial orthogonal polyhnoimals
%
% p = mpoly_eval(x, alpha, recurrence, {d = zeros([1 dim])} )
%
%   Evaluates tensor-product polynomials that are orthonormal with respect to 
%   a tensor-product probability measure. The dimension dim is equal to
%   both size(x,2), and size(alphas,2), and size(d, 2).
%
%   Each row of the matrix x is a point in dim-dimensional space, and each row
%   of the matrix alpha is a dim-dimensional multi-index. Each row of d is also
%   a multi-index specifying a partial derivative to be computed.
%
%   The input recurrence is a function handle with the syntax
%
%     [a,b] = recurrence(N)
% 
%   returning the first N recurrence coefficients for the univariate measure.
%   (So the tensor-product measure is defined as the dim-fold product of this
%   measure.)
%
%     Input
%     -----------------
%     x           (M x dim) array, each row is a point on [-1,1]^d
%     alpha       (N x dim) array, each row is a dim-dimensional multi-index
%     recurrence  function handle with syntax [a,b] = recurrence(N) for scalar integer N
%     d           (P x dim) array, each row is a dim-dimensional multi-index
%
%     Output
%     -----------------
%     p         (M x N x P) array, rows correspond to entries of x

[M,dim] = size(x);

if nargin < 4
  d = zeros([1 dim]);
else
  assert(size(d,2) == dim, 'Inputs x and d must have the same number of columns');
end

N = size(alpha,1);
P = size(d, 1);

assert(size(alpha,2)==dim, 'Inputs x and alpha must have the same number of columns');

[a, b] = recurrence(max(alpha(:)) + 1);
p = ones([M N P])/sqrt(b(1)^dim);

% Find which alphas are positive
a0 = alpha > 0;

for qd = 1:P
  
  for qdim = 1:dim
    temp = poly_eval(a, b, x(:,qdim), max(alpha(:,qdim)), d(qd,qdim) );

    if d(qd,qdim) > 0
      p(:,:,qd) = p(:,:,qd) .* temp(:, alpha(:,qdim)+1) * sqrt(b(1));
    else
      % Only update columns whose indices are positive
      p(:,a0(:,qdim),qd) = p(:,a0(:,qdim),qd) .* temp(:, alpha(a0(:,qdim),qdim)+1) * sqrt(b(1));
    end

  end

end
