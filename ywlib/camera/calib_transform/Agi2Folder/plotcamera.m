function [] = plotcamera( varargin )
defult_Transfrom = [eye(3) [0; 0; 0]];
defaults = struct('Transform', defult_Transfrom, 'Color', [0 0 1], ...
    'Scale', [1 1 1], 'Thickness', 1,'Fill', 0, 'Righthand', 1);  %define default values

params = struct(varargin{:});

for f = fieldnames(defaults)',
    if ~isfield(params, f{1}),
        params.(f{1}) = defaults.(f{1});
    end
end

transform = params.Transform;
color     = params.Color;
scale     = params.Scale;
isfill    = params.Fill;
thickness = params.Thickness;
isRighthand = params.Righthand;

if isRighthand
    z = 1;
else
    z = -1;
end
vertices = [0 0 0;-0.5 0.5 z;0.5 0.5 z;0.5 -0.5 z;-0.5 -0.5 z];
axis_x = [1 0 0];
axis_y = [0 1 0];

vertices(:, 1) = vertices(:, 1) * scale(1);
vertices(:, 2) = vertices(:, 2) * scale(2);
vertices(:, 3) = vertices(:, 3) * scale(3);

axis_x = axis_x .* scale;
axis_y = axis_y .* scale;

vertices = transform(1:3, 1:3) * vertices' + repmat(transform(:, 4), [1 5]);
vertices = vertices';

axis_x = (transform(1:3, 1:3) * axis_x' + transform(:, 4))';
axis_y = (transform(1:3, 1:3) * axis_y' + transform(:, 4))';

for i = 1:5
    if i == 1
        for nex = 2:5
            hold on
            line([vertices(i, 1); vertices(nex, 1)], ...
                [vertices(i, 2); vertices(nex, 2)], ...
                [vertices(i, 3); vertices(nex, 3)], ...
                'LineWidth', thickness, ...
                'Color', color);
        end
        
        %         if isfill
        %             hold on
        %             fill3([vertices(i, 1); vertices(nex, 1)], ...
        %                 [vertices(i, 2); vertices(nex, 2)], ...
        %                 [vertices(i, 3); vertices(nex, 3)], ...
        %                 'Color', color);
        %         end
    else
        nex = i + 1;
        if nex > 5; nex = 2; end
        
        hold on
        line([vertices(i, 1); vertices(nex, 1)], ...
            [vertices(i, 2); vertices(nex, 2)], ...
            [vertices(i, 3); vertices(nex, 3)], ...
            'LineWidth', thickness, ...
            'Color', color);
    end
end

n_axis_x = axis_x - vertices(1, :);
n_axis_y = axis_y - vertices(1, :);

quiver3(vertices(1, 1), vertices(1, 2), vertices(1, 3), ...
    n_axis_x(1), n_axis_x(2), n_axis_x(3), ...
    'Color', [1 0 0]);

quiver3(vertices(1, 1), vertices(1, 2), vertices(1, 3), ...
    n_axis_y(1), n_axis_y(2), n_axis_y(3), ...
    'Color', [0 1 0]);

% if isfill
%     hold on
%     fill3(vertices(2:5, :), vertices(2:5, :), vertices(2:5, :));
% end

end