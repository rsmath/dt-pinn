function[a,b] = jacobi_recurrence(N, alph, bet)
% [a,b] = jacobi_recurrence(N, alph, bet)
%
%     Returns the first N three-recurrence coefficients for the Jacobi
%     polynomial family with parameters alph, bet.

N = max(N(:));
n = (1:N).' - 1;

a = (bet^2-alph^2)*ones(size(n));
b = ones(size(n));

% Initial conditions:
flags0 = (n==0);
if any(flags0)
  a(flags0) = (bet-alph)/(alph+bet+2);

  b(flags0) = exp( (alph + bet + 1) * log(2) + gammaln(alph + 1) + gammaln(bet+1) - gammaln(alph + bet + 2));

end

flags1 = (n==1);
a(flags1) = a(flags1)./((2+alph+bet)*(4+alph+bet));
b(flags1) = 4*(1+alph)*(1+bet)/((2+alph+bet)^2*(3+alph+bet)); 

flags = not(flags0 | flags1);
a(flags) = a(flags)./((2*n(flags)+alph+bet).*(2*n(flags)+alph+bet+2));
b(flags) = 4*n(flags).*(n(flags)+alph).*(n(flags)+bet).*(n(flags)+alph+bet);
b(flags) = b(flags)./((2*n(flags)+alph+bet).^2.*...
                     (2*n(flags)+alph+bet+1).*(2*n(flags)+alph+bet-1));
