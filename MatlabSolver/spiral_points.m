function p = spiral_points ( r, center, n )
  p = zeros(n,3);

  for i = 1 : n

    cosphi = ( -(n-i)+ (i-1))/(n-1);

    sinphi = sqrt ( 1.0 - cosphi^2 );

    if ( i == 1 || i == n )
      theta = 0.0;
    else
      theta = theta + 3.6 / ( sinphi * sqrt ( n ) );
      theta = mod ( theta, 2.0 * pi );
    end

    p(i,1) = center(1,1) + r * sinphi * cos ( theta );
    p(i,2) = center(1,2) + r * sinphi * sin ( theta );
    p(i,3) = center(1,3) + r * cosphi;
  end
end