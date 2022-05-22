function[a,b] = chebyshev_recurrence(N)
% chebyshev_recurrence -- Three-term recurrence coefficients for Chebyshev polynomials
%
% [a,b] = chebyshev_recurrence(N)
%
%     Returns the first N recurrence coefficients for the (orthonormal)
%     Chebyshev polynomial family associated to the arcsine probability measure
%     on [-1,1].

% Recurrence coefficients: Special case of Jacobi polynomials with alpha = beta = 0.5
a = zeros([N 1]);
b = ones([N 1]);

if N > 1
  b(2) = 1/2;
  b(3:end) = 1/4;
end
