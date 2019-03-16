clear; clc; close all;
% file path
filepath = "../dataset/fixed/";

for nimg = 1:1:1134
        filename = num2str(nimg);
        depth = imread(filepath + 'dep' + filename + '.png');
        % extract U map
        downsample = 10;
        tic;
        [U_map, U_edges] = extract_u_map(depth, downsample);
        % contiguous lines
        [poi,seg_name] = extract_contiguous_lines(U_map,depth,downsample);
        % group lines and extract bounding boxes
        bb = [];
        for seg_idx = 1:length(seg_name)
           [seg_rows,seg_cols] = find(poi == seg_idx);
           tlr = min(seg_rows); tlc = min(seg_cols); % top-left
           brr = max(seg_rows); brc = max(seg_cols); % bottom-right
           bb = [bb,[tlr;tlc;brr;brc]];
        end
        toc;
        % extract object's height
        for i = 1:size(bb,2)
            V_map = sum(depth(:,bb(2,i):bb(4,i)),2);
            figure(2)
            imagesc(V_map);
        end
        % visualization
        figure(1);
        subplot(5,1,1)
        imagesc(U_map); title('U map','FontSize',10); axis off;
        for i = 1:size(bb,2)
           rectangle('Position',[bb(2,i),bb(1,i),bb(4,i) - bb(2,i),2 * (bb(3,i)-bb(1,i))],'LineWidth',2,'EdgeColor','r'); 
        end
        subplot(5,1,[2,3,4,5])
        imagesc(depth); axis normal; axis off; title('depth map','FontSize',10);
    pause(0);
        
%     % GIF
%     h = figure(1);
%     frame = getframe(h); 
%     im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
%     ts = 0.05;
%     % Write to the GIF File 
%     if nimg == 1 
%       imwrite(imind,cm,'../img/move_result.gif','gif', 'Loopcount',inf,'DelayTime',ts); 
%     elseif rem(nimg, 5) == 1
%       imwrite(imind,cm,'../img/move_result.gif','gif','WriteMode','append','DelayTime',ts); 
%     end
end