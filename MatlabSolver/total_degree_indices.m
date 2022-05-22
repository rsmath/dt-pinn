function[a] = total_degree_indices(d, k)
% total_degree_indices -- Total degree indices
% 
% [a] = total_degree_indices(d, k)
%
%     Computes all multi-indices of degree k or less in d dimensions.
%
%     Computes (0-based) array indexing from (1-based) linear indexing. The
%     output a is a size(n,1) x dim matrix, where each row corresponds to the
%     0-based array indexing for the given linear index n.
%
%     Example:
%          k = 2 , d = 2 ----> [0 , 0
%                               1 , 0
%                               0 , 1
%                               2 , 0
%                               1 , 1
%                               0 , 2]
%
%     Input
%     -----------------
%     n         length-N vector, entries are non-negative integers
%     d         scalar, positive integer
%
%     Output
%     -----------------
%     a         (N x d) array, row k is the multi-index corresponding to n(k)
%

n = (1:nchoosek(d+k, d));

n = n - 1;

n = n(:);
n = n+1;
n_max = max(n);

% Find the largest polynomial order we'll need:
N = 0;
while nchoosek(N+d,d)<n_max
  N = N + 1;
end

% Generate all the tuples up to max(n) and then pluck out the ones we want
a = zeros([nchoosek(N+d,d), d]);

row_id = 1;
a(row_id,:) = zeros([1 d]);
row_id = 2;

if N == 0
  a = zeros([size(n,1) d]);
else
  for q = 1:N
    current_row = zeros([1 d]);
    current_row(1) = q;
    a(row_id,:) = current_row;
    row_id = row_id + 1;

    % "The traveling ones-man method"
    onesman_home = 1;
    onesman_location = 1;

    finished = false;

    while not(finished)
      onesman_pilgrimage();
    end
  end

  a = a(n,:);
end

function[] = onesman_pilgrimage()
  while onesman_location < d
    onesman_location = onesman_location + 1;
    current_row(onesman_location-1) = current_row(onesman_location-1) - 1;
    current_row(onesman_location) = current_row(onesman_location) + 1;
    a(row_id,:) = current_row;
    row_id = row_id + 1;
  end

  if onesman_home + 1 == d 
    % Then make all the other onesman in column d-1 travel as well
    while current_row(onesman_home)>0
      current_row(end) = current_row(end) + 1;
      current_row(end-1) = current_row(end-1) - 1;
      a(row_id,:) = current_row;
      row_id = row_id + 1;
    end
  end

  if current_row(end)==q
    finished = true;
    return % done
  end

  % Now update new home for (next) onesman
  % There must exist a zero in some column; find the last consecutive one from
  % the right

  columns = find(current_row, 2, 'last');
  current_row(columns(1)) = current_row(columns(1)) - 1;
  current_row(columns(1)+1) = current_row(end) + 1;
  current_row(end) = 0;
  a(row_id,:) = current_row;
  row_id = row_id + 1;

  onesman_home = columns(1)+1;
  onesman_location = columns(1)+1;
end

end
