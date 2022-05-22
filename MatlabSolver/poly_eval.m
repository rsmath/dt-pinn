function[p] = poly_eval(a, b, x, N, d)
% poly_eval -- Evaluates orthogonal polynomials
%
% p = poly_eval(a, b, x, N, {d=0} )
%   Uses the recurrence coefficients a and b to evaluate the d'th derivative of
%   p_n(x), where p_n(x) is the degree-n orthonormal polynomial associated with
%   the recurrence coefficients a, b (with positive leading coefficient).
%
%   The p_n satisfy the recurrences
%
%     sqrt(b_{n+1}) p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}
%
%   With the input arrays a and b, we have a_n = a(n+1), b_n = b(n+1). Note
%   that b_0 = b(1) is only used to initialize p_0.
%
%   The output matrix p has size numel(x) x (N+1), and hence the first N+1 (up
%   to degree N) polynomials are evaluated.
%
%   Inputs:
%       x : array of doubles
%       N : positive integer, N - 1 < length(a) == length(b)
%       a : array of recurrence coefficients
%       b : array of reucrrence coefficients
%       d : non-negative integer (default: 0)

if nargin < 5
  d = 0;
else
  assert(d >= 0)
end

nx = numel(x);

assert(N >= 0);

assert(N <= length(a));
assert(N <= length(b));

p = zeros([nx N+1]);

% Flatten x
xf = x(:);

% To initialize r, we need p_0 and p_1
p(:,1) = 1/sqrt(b(1)) * ones([nx 1]);
if N > 0
  p(:,2) = 1/sqrt(b(2)) * (xf - a(1)).*p(:,1);
end

for q = 2:N
  % Derived from three-term recurrence
  p(:,q+1) = (xf - a(q)).*p(:,q) - sqrt(b(q)).*p(:,q-1);
  p(:,q+1) = 1/sqrt(b(q+1)) * p(:,q+1);
end

%% Evaluate derivatives if necessary
if d == 0
  return
end

for qd = 1:d
  % Assume p stores the order (qd-1) derivative
  pd = zeros(size(p));

  for q = qd:N

    if q == qd
      pd(:,q+1) = exp( gammaln(qd+1) - 0.5*sum( log(b(1:q+1)) ) );
    else
      pd(:,q+1) = (xf - a(q)).*pd(:,q) - sqrt(b(q)).*pd(:,q-1) + qd*p(:,q);
      pd(:,q+1) = 1/sqrt(b(q+1)) * pd(:,q+1);
    end

  end

  p = pd;

end
