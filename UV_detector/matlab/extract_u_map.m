function [U_map, U_edges] = extract_u_map(depth,downsample)
    row = size(depth,1);
    col = size(depth,2);
    % U map
    min_dist = 10;
    max_dist = 5000;
    U_edges = linspace(min_dist,max_dist,row/downsample + 1);
    U_map = uint16(zeros(numel(U_edges) - 1, col));
    for r = 1:col
        [N,~] = histcounts(depth(:,r), U_edges);
        U_map(:,r) = N;
    end
end