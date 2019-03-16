function [poi,seg_name] = extract_contiguous_lines(U_map,depth,downsample)
   % contiguous lines
    poi = uint8(zeros(size(U_map)));
    u_max = 2 * downsample;
    sum_line = 0;
    max_line = 0;
    length_line = 0;
    seg_idx = 0;
    seg_name = [];
    for row = 1:1:size(U_map,1)
        for col = 1:1:size(U_map,2)
           if U_map(row, col) >= u_max
               length_line = length_line + 1;
               sum_line = sum_line + U_map(row, col);
               if U_map(row, col) > max_line
                  max_line = U_map(row, col); 
               end
           end
           % new line
           if U_map(row, col) < u_max || col == size(U_map,2)
               if(col == size(U_map,2))
                  col = col + 1; 
               end
               if length_line >= 30
                if sum_line > 2 * max_line
                    % first row
                    if row == 1
                        seg_idx = seg_idx + 1;
                        seg_name = [seg_name, seg_idx];
                        poi(row, col - length_line + 1:col) = seg_idx;
                    else
                        % no parent
                        if max(poi(row - 1, col - length_line:col - 1)) == 0;
                            seg_idx = seg_idx + 1;
                            seg_name = [seg_name, seg_idx];
                            poi(row, col - length_line:col - 1) = seg_idx;
                        % have parent
                        else
                            poi(row, col - length_line:col - 1) = max(poi(row - 1, col - length_line:col-1));
                            % merge all parents
                            for p_col = col - length_line:col - 1
                               if poi(row - 1, p_col) ~= 0
                                  [prows,pcols]= find(poi == poi(row - 1, p_col));
%                                   if seg_name ~= max(poi(row - 1, col - length_line:col-1))
%                                     [seg_name_idx] = find(seg_name == poi(row - 1, p_col));
%                                     seg_name(seg_name_idx) = [];
%                                   end
                                  for i = 1:size(prows)
                                     poi(prows(i),pcols(i)) = max(poi(row - 1, col - length_line:col-1));
                                     if prows(i) == row - 1 && pcols(i) > p_col
                                         p_col = pcols(i);
                                     end
                                  end
                               end
                            end
                        end
                    end
                end
               end
                sum_line = 0;
                max_line = 0;
                length_line = 0;
           end
        end
    end
end