function [point_type] = ClassifyPoints(X,geom_models,h)
%CLASSIFYPOINTS Classify a set of grid points as inside/outside/forcing.
%  This classification is done with respect to a set of embedded
%  boundaries described by point samples and normals, and bounding boxes.

%% First, build PCA bounding boxes for each object
Nobj = size(geom_models,1);
Bb = cell(Nobj,1);
sides = cell(Nobj,1);
for k=1:Nobj
    [sides{k},Bb{k}] = GetBoundingBoxSides(geom_models(k,1).xe + geom_models(k,1).nre*0.75*h);
end

%% Then, test each node against the PCA bounding boxes. If outside,
%%mark as fluid (0). If inside, test against the specific object's normals.
%%If inside the object, mark as solid (1).
point_type = zeros(length(X),2); %points are fluid by default
goto_next_pt=0;
for j=1:length(X)
    p = X(j,:);
    for k=1:Nobj
        %in_box_flag = IsPointInBox(p,Bb{k});
        %if in_box_flag==1 %inside the bounding box
            in_obj_flag = IsPointInObj(p,geom_models(k,1).xe, geom_models(k,1).nre,h);            
            if in_obj_flag==1     
                point_type(j,1) = 1; %Mark point as solid
                point_type(j,2) = k; %Say it belongs to object k
                goto_next_pt = 1;
            end
        %else %outside the bounding box
        %    continue; %not in this object, test against next object
       % end
        if goto_next_pt == 1 %a point can only be inside one object, stop testing
            goto_next_pt = 0;
            break; %stops testing point against objects
        end
    end    
end

end

