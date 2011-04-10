function nimg = wang(img)

if(ischar(img)), img = imread(img);
end
%figure(1), imshow(img);

% Step 1
if(ndims(img) == 3), img = rgb2gray(img);
end
img = double(img);

[height width] = size(img);

adjustment = mean([height/300 width/400]);
one_hundred = 100; %100*adjustment
ratio = 6.2; %4
ten_of_height = 12; %10*adjustment
five_rows_above = 5*adjustment; %5*adjustment

% Step 2
[ll lh hl hh] = dwt2(img,'db1','mode','sym');
[height_hl width_hl] = size(hl);

% Step 3
abs_hl = abs(hl);
hl_norm = abs_hl ./ max(abs_hl(:));
hl_bin = hl_norm >= graythresh(hl_norm);

avg_height_v = [];
[l num] = bwlabel(hl_bin, 8);
hl_bin = double(hl_bin);
for i = 1 : num,
    [rows cols] = find(l == i);
    line_height = max(rows) - min(rows) + 1;
    indexes = sub2ind([height_hl width_hl], rows, cols);
    if(line_height > one_hundred), hl_bin(indexes) = 0;
    else
        hl_bin(indexes) = line_height;
        avg_height_v(end+1) = line_height;
    end
end

avg_height_v = mean(avg_height_v);

hl_vi_bin = hl_bin > avg_height_v;
hl_vi = hl .* hl_vi_bin;

% Step 4
abs_lh = abs(lh);
avg_lh = mean(abs_lh(:));
p = abs_lh .* (abs_lh >= avg_lh);
p_bin = p ~= 0;

% Step 5
bs = floor(1/5 * width_hl);
lines = zeros([height_hl width_hl]);

transition = double(hl_vi_bin);
transition = transition(:, 1 : end-1) - transition(:, 2 : end);
transition = transition ~= 0;
transition = [transition zeros([height_hl 1])];

for i = height_hl : -1 : 1,
    for j = 1 : width_hl-bs, lines(i, j) = sum(transition(i, j : j+bs));
    end
end

lines = lines .* hl_vi_bin;
for i = height_hl : -1 : 1, lines(i, lines(i, :) ~= max(lines(i, :))) = 0;
end

for i = height_hl : -1 : five_rows_above+1,
    for j = 1 : width_hl-bs,
        if(lines(i, j)),
            se = ones([1 floor(bs/2)]);
            lines(i, j) = lines(i, j) && max(max(imopen(p_bin(i-five_rows_above : i-1, j : j+bs), se)));
        end
    end
end

[l num] = bwlabel(lines, 8);
candidates_x = [];
candidates_y = [];
for k = 1 : num,
    [rows cols] = find(l == k);
    candidates_y(end+1) = min(rows);
    candidates_x(end+1) = min(cols);
end

% Step 6, 7, 8, and 9
nimg = zeros([height width]);
for k = 1 : length(candidates_x)
    sy = candidates_y(k);
    sx = candidates_x(k);
    ey = min([sy + ten_of_height, height_hl]);
    ex = min([sx + floor(ten_of_height * ratio), width_hl]);
    candidate_width = ex - sx;

    % Step 7
    window_width = floor(candidate_width/3);

    new_sx = sx;
    max_value = 0;
    for j = sx : sx+window_width,
        value = sum(sum(transition(sy : ey, j : j+window_width)));
        if(value > max_value),
            max_value = value;
            new_sx = j;
        end
    end

    new_ex = ex;
    max_value = 0;
    for j = ex : -1 : ex-window_width,
        value = sum(sum(transition(sy : ey, j-window_width : j)));
        if(value > max_value),
            max_value = value;
            new_ex = j;
        end
    end
    
    % Step 8
    new_sy = sy;
    for i = sy : -1 : 1,
        if(~sum(hl_vi(i, sx : ex))), new_sy = i; break;
        end
    end
    
    new_ey = ey;
    for i = ey : height_hl,
        if(~sum(hl_vi(i, sx : ex))), new_ey = i; break;
        end
    end
    
    sx = new_sx;
    ex = new_ex;
    sy = new_sy;
    ey = new_ey;
    
    % Step 9
    sx = 2*sx;
    ex = 2*ex;
    sy = 2*sy;
    ey = 2*ey;
    
    img_candidate = zeros([height width]);
    img_candidate(sy : ey, sx : ex) = 1;

    nimg = nimg + img_candidate;
end
%figure(2); imshow(nimg);
