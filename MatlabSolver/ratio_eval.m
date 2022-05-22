function[r] = ratio_eval(a, b, x, N)
% ratio_eval -- Evaluates ratios between successive orthogonal polynomials
%
% r = ratio_eval(a, b, x, N)
%   Uses the recurrence coefficients a and b to evaluate the ratio r at
%   locations x, of orders n = 1, ..., N. A ratio r_n(x) of order n is defined
%   by
%
%       r_n(x) = p_n(x) / p_{n-1}(x),
%
%   where p_n(x) is the degree-n orthonormal polynomial associated with the
%   recurrence coefficients a, b (with positive leading coefficient).
%
%   The p_n and r_n satisfy the recurrences
%
%     sqrt(b_{n+1}) p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}
%     sqrt(b_{n+1}) r_{n+1} = (x - a_n)  - sqrt(b_n) / r_n
%
%   With the input arrays a and b, we have a_n = a(n+1), b_n = b(n+1). Note
%   that b_0 = b(1) is only used to initialize p_0.
%
%   For the Nevai class, we expect a_n ---> 0, b_n ---> 1/4
%
%   The output matrix r has size numel(x) x N
%
%   Inputs:
%       x : array of doubles
%       N : positive integer, N - 1 <= length(a) == length(b)
%       a : array of recurrence coefficients
%       b : array of reucrrence coefficients

nx = numel(x);

assert(N > 0);

assert(N < length(a));
assert(N < length(b));

r = zeros([nx N]);

% Flatten x
xf = x(:);

% To initialize r, we need p_0 and p_1
p0 = 1/sqrt(b(1)) * ones([nx 1]);
p1 = 1/sqrt(b(2)) * (xf - a(1)).*p0;

r1 = p1./p0;
r(:,1) = r1;

for q = 2:N
  % Derived from three-term recurrence
  r2 = (xf - a(q)) - sqrt(b(q)) ./ r1;
  r1 = 1/sqrt(b(q+1)) * r2;

  r(:,q) = r1;
end

if nx == 1
  r = r(:);
end
