function [ res ] = IsPointInBox( point,p )
% Figures out if a point is inside a box
   
A = p(1,:);
B = p(2,:);
C = p(3,:);
D = p(4,:);

AB = B-A;
AD = D-A;
AP = point-A;

dotABAP = dot(AB, AP);
dotAPAD = dot(AP, AD);

res = (0 <= dotABAP && dotABAP <= dot(AB, AB)) && (0 <= dotAPAD && dotAPAD <= dot(AD, AD));
end

