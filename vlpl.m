function nimg = vlpl(img, MIN_HEIGHT_CHAR, MAX_HEIGHT_CHAR, ...
    MIN_WIDTH_VLP, MAX_DISTANCE_CHARS, MAX_DISTANCE_PARTS, ...
    HEIGHT_MEAN_FILTER, WIDTH_MEAN_FILTER, SIZE_DILATE, ...
    SIZE_DILATE_FINAL, DECISION_AMONG_CANDIDATES)

if(ischar(img)), img = imread(img);
end
%figure(1); imshow(img);

if(nargin == 1),
    MIN_HEIGHT_CHAR = 11;
    MAX_HEIGHT_CHAR = 43;
    MIN_WIDTH_VLP = 72;
    MAX_DISTANCE_CHARS = 20;
    MAX_DISTANCE_PARTS = 5;
    HEIGHT_MEAN_FILTER = 6;
    WIDTH_MEAN_FILTER = 25;
    SIZE_DILATE = 5;
    SIZE_DILATE_FINAL = 7;
    DECISION_AMONG_CANDIDATES = true;
end
MIN_HEIGHT_VLP = MIN_HEIGHT_CHAR;

if(ndims(img) == 3), img = rgb2gray(img);
end
img = im2double(img);

[height width] = size(img);

for ind = 1 : 2,
    if(ind == 2), % Hitogram equalization.
        hist = imhist(img)/(height*width);
        hist = cumsum(hist);
        img = hist(im2uint8(img)+1);
    end

    % Horizontal gradient
    img_proc = abs(filter2(fspecial('sobel')', img));

    % Mean filter
    op = fspecial('average', [HEIGHT_MEAN_FILTER WIDTH_MEAN_FILTER]);
    img_proc = filter2(op, img_proc);

    % Darkening small salient regions
    se = ones([MIN_HEIGHT_CHAR 1]);
    img_proc = imopen(img_proc, se);

    % Merging vertical saliences
    se = ones([1 MAX_DISTANCE_CHARS]);
    img_proc = imclose(img_proc, se);

    % Darkening big salient regions
    se = ones([MAX_HEIGHT_CHAR 1]);
    img_with_high_saliences = imopen(img_proc, se);
    img_proc = img_proc - img_with_high_saliences;

    % Joining parts of the VLP
    se = ones([1 MAX_DISTANCE_PARTS]);
    img_proc = imclose(img_proc, se);

    % Erosion and dilation
    se = ones([1 2*WIDTH_MEAN_FILTER]);
    img_proc = imerode(img_proc, se);
    se = ones([1 uint8(1.5*WIDTH_MEAN_FILTER)]);
    img_proc = imdilate(img_proc, se);
    
    img_filtering = img_proc;

    % Binarization
    img_proc = img_proc >= graythresh(img_proc); % Otsu

    % Removing non-VLP shape objects
    [l num] = bwlabel(img_proc, 4);
    img_proc(:,:) = 0;
    for i = 1 : num,
        [rows cols] = find(l == i);
        row_min = min(rows); row_max = max(rows);
        col_min = min(cols); col_max = max(cols);
        object_height = row_max-row_min+1;
        object_width = col_max-col_min+1;
        if(object_height >= object_width), continue;
        elseif(row_min == 1 || row_max == height || ...
                col_min == 1 || col_max == width), continue;
        elseif(object_height < MIN_HEIGHT_VLP || ...
                object_width < MIN_WIDTH_VLP), continue;
        end
        img_proc(row_min : row_max, col_min : col_max) = 1;
    end

    % Removing candidates in intersection
    [l num] = bwlabel(img_proc, 4);
    img_proc(:,:) = 0;
    for i = 1 : num,
        [rows cols] = find(l == i);
        row_min = min(rows); row_max = max(rows);
        col_min = min(cols); col_max = max(cols);
        object_area = (row_max-row_min+1)*(col_max-col_min+1);
        if(length(find(l == i)) < object_area), continue;
        end
        img_proc(row_min : row_max, col_min : col_max) = 1;
    end

    % Adjustment to the VLP
    se = ones(SIZE_DILATE);
    [l num] = bwlabel(img_proc, 4);
    imgs = img_proc;
    for i = 1 : num, imgs(:, :, i) = imdilate(l == i, se);
    end
    
    max_value = 0;
    img_max_value = 0;
    nimg = zeros([height width]);
    for k = 1 : num,
        [rows cols] = find(imgs(:, :, k));
        row_min = min(rows); row_max = max(rows);
        col_min = min(cols); col_max = max(cols);
        img_vlp = img_filtering(row_min : row_max, col_min : col_max);
        img_vlp = img_vlp >= graythresh(img_vlp);
        se = ones([1 uint8(0.25*WIDTH_MEAN_FILTER)]);
        img_vlp = imerode(img_vlp, se);
        se = ones([uint8(0.5*HEIGHT_MEAN_FILTER) ...
            uint8(0.5*WIDTH_MEAN_FILTER)]);
        img_vlp = imdilate(img_vlp, se);
        tmp = zeros([height width]);
        tmp(row_min : row_max, col_min : col_max) = img_vlp;
        new_row_min = row_min; new_row_max = row_max;
        new_col_min = col_min; new_col_max = col_max;
        for i = row_min : row_max,
            if(sum(tmp(i, col_min : col_max))), break;
            else new_row_min = new_row_min+1;
            end
        end
        for i = row_max : -1 : row_min,
            if(sum(tmp(i, col_min : col_max))), break;
            else new_row_max = new_row_max-1;
            end
        end
        for j = col_min : col_max,
            if(sum(tmp(row_min : row_max, j))), break;
            else new_col_min = new_col_min+1;
            end
        end
        for j = col_max : -1 : col_min,
            if(sum(tmp(row_min : row_max, j))), break;
            else new_col_max = new_col_max-1;
            end
        end
        
        object_height = new_row_max-new_row_min+1;
        object_width = new_col_max-new_col_min+1;
        if(object_height < MIN_HEIGHT_VLP || ...
                object_width < MIN_WIDTH_VLP || ...
                object_height >= object_width),
            if(num == 1),
                img_proc = imgs(:, :, 1);
                se = ones(SIZE_DILATE_FINAL);
                img_proc = imdilate(img_proc, se);
                nimg = img_proc;
                %figure(2); imshow(nimg);
                return;
            else continue;
            end
        end
        imgs(row_min : new_row_min-1, col_min : col_max, k) = 0;
        imgs(new_row_max+1 : row_max, col_min : col_max, k) = 0;
        imgs(row_min : row_max, col_min : new_col_min-1, k) = 0;
        imgs(row_min : row_max, new_col_max+1 : col_max, k) = 0;

        % Choosing the VLP
        [valid_rows valid_cols] = find(tmp(new_row_min : new_row_max, ...
            new_col_min : new_col_max));
        valid_rows = valid_rows + (new_row_min-1);
        valid_cols = valid_cols + (new_col_min-1);
        inds = sub2ind([height width], valid_rows, valid_cols);
        values = img_filtering(inds);
        mu = mean(values);
        sd = std(values);
        cv = sd/mu;
        value = mu/cv;
        if(value > max_value),
            max_value = value;
            img_max_value = k;
        end
        % This part is only useful when DECISION_AMONG_CANDIDATES is false
        se = ones(SIZE_DILATE_FINAL);
        tmp(new_row_min : new_row_max, new_col_min : new_col_max) = 1;
        nimg = nimg + imdilate(tmp, se);
    end

    if(img_max_value == 0 && ind == 1), continue;
    end
    if(img_max_value), img_proc = imgs(:, :, img_max_value);
    end
    if(DECISION_AMONG_CANDIDATES),
        se = ones(SIZE_DILATE_FINAL);
        nimg = imdilate(img_proc, se);
    end
    %figure(2); imshow(nimg);
    return;
end
