function nimg = vargas(img, ALPHA, MAX_HEIGHT_PIV, MIN_HEIGHT_PIV, ...
    MIN_WIDTH_PIV, MAX_WIDTH_PIV, RATIO_CHAR, K_BETA_ROWS, K_GAMA_ROWS, ...
    K_BETA_COLS, K_GAMA_COLS, ALPHA_1, ALPHA_2, K_BCKGRD)

if(ischar(img)), img = imread(img);
end
%figure(1); imshow(img);

if(nargin == 1)
    ALPHA = 5;
    ALPHA = ALPHA/100;
    MAX_HEIGHT_PIV = 70;
    MIN_HEIGHT_PIV = 12;
    MIN_WIDTH_PIV = 36;
    MAX_WIDTH_PIV = 220;
    RATIO_CHAR = 50/77;
    K_BETA_ROWS = 0.25; %0.15
    K_GAMA_ROWS = 0.44; %0.3
    K_BETA_COLS = 0.27; %0.2
    K_GAMA_COLS = 0.25; %0.25
    ALPHA_1 = 0.47; %0.49
    ALPHA_2 = 0.15; %0.31
    K_BCKGRD = 5.9;
end

if(ndims(img) == 3), img = rgb2gray(img);
end
img = im2uint8(img);

[height , width] = size(img);

% Vertical-edge image
img_vertical_edge = abs(filter2(fspecial('sobel')', img));

% Binarization of vertical-edge image
pixels_number = height * width;
[histogram bins] = histog(uint16(img_vertical_edge));
cumulative_histogram = cumsum(histogram(end : -1 : 1));

i = find(cumulative_histogram./pixels_number > ALPHA, 1, 'first');
threshold = bins - i + 1;
img_binary = img_vertical_edge >= threshold;

% Candidate extraction using vertical-edge density
h = ceil((MAX_HEIGHT_PIV + MIN_HEIGHT_PIV)*0.85 / 2);
w = ceil(h * RATIO_CHAR * 0.7);
img_density = filter2(fspecial('average' , [h w]), img_binary);

% Find candidates
candidates_upp = [];
candidates_bot = [];
candidates_left = [];
candidates_right = [];
candidates_center_rows = [];
candidates_center_cols = [];

% Find candidates rows
v = zeros([height 1]);
for i = 1 : height, v(i) = var(img_density(i , :));
end

BETA_ROWS = K_BETA_ROWS * max(v);
[r_final r_upp_final r_bot_final] = find_candidate(v, BETA_ROWS, ...
    K_GAMA_ROWS, MIN_HEIGHT_PIV, MAX_HEIGHT_PIV);

nimg = zeros([height width]);
length_r_final = length(r_final);
if(~length_r_final), return;
end

% Find candidates cols
v = zeros([length_r_final width]);
for i = 1 : length_r_final,
    for j = 1 : width, v(i, j) = var(img_density(r_upp_final(i) : ...
            r_bot_final(i), j));
    end
end

for i = 1 : length_r_final,
    BETA_COLS = K_BETA_COLS * max(v(i, :));
    [c_final c_left_final c_right_final] = find_candidate(v(i, :), ...
        BETA_COLS, K_GAMA_COLS, MIN_WIDTH_PIV, MAX_WIDTH_PIV);

    for j = 1 : length(c_final),
        candidates_upp = [candidates_upp r_upp_final(i)];
        candidates_bot = [candidates_bot r_bot_final(i)];
        candidates_left = [candidates_left c_left_final(j)];
        candidates_right = [candidates_right c_right_final(j)];
        candidates_center_rows = [candidates_center_rows r_final(i)];
        candidates_center_cols = [candidates_center_cols c_final(j)];
    end
end

% Two-step region growing
nimg = zeros([height width]);
for l = 1 : length(candidates_upp),
    maximum = max(max(img_density(candidates_upp(l) : candidates_bot(l), ...
        candidates_left(l) : candidates_right(l))));
    [row col] = find(img_density(candidates_upp(l) : candidates_bot(l), ...
        candidates_left(l) : candidates_right(l)) == maximum, 1, 'first');

    row = row + candidates_upp(l) - 1;
    col = col + candidates_left(l) - 1;

    img_region_upper_density = seeded_region_growing(img_density, ...
        row, col, ALPHA_1 * img_density(row , col));

    hist_region_growing = histog(img(img_region_upper_density), 256);
    N = sum(img_region_upper_density(:));
    E = sum((hist_region_growing ./ N) .^ 2);
    gama = find(cumsum((hist_region_growing ./ N) .^2) >= 0.75*E, ...
        1, 'first');
    mean_value = mean(img(img_region_upper_density));

    tmp = img(img_region_upper_density);
    if(gama > mean_value), tmp = tmp(tmp >= gama);
    else tmp = tmp(tmp < gama);
    end
    mean_bckgrd = mean(tmp);
    std_bckgrd = std(tmp);

    img_region_lower_density = seeded_region_growing(img_density, ...
        row, col, ALPHA_2 * img_density(row , col));

    img_candidate = zeros([height width]);
    img_candidate(img_region_lower_density) = img(img_region_lower_density);
    img_candidate(img_candidate < mean_bckgrd - K_BCKGRD*std_bckgrd) = 0;
    img_candidate(img_candidate > mean_bckgrd + K_BCKGRD*std_bckgrd) = 0;

    labels = bwlabel(img_candidate, 8);
    img_candidate = zeros([height width]);
    uniques = unique(labels(img_region_upper_density));
    for i = 1 : length(uniques),
        if(uniques(i)), img_candidate = img_candidate | labels == uniques(i);
        end
    end

    [rows cols] = find(img_candidate > 0);
    img_candidate(min(rows) : max(rows), min(cols) : max(cols)) = 1;

    nimg = nimg + img_candidate;
end

%%
function nimg = seeded_region_growing(img_density, seed_row, ...
    seed_col, threshold)

labels = bwlabel(img_density > threshold, 8);
nimg = labels == labels(seed_row, seed_col);

%%
function [hist, bins] = histog(img , num_bins)

img = img(:);
if(nargin == 1), bins = max(img) + 1;
else bins = num_bins;
end
hist = zeros([bins 1]);

for i = 1 : length(img), hist(img(i)+1) = hist(img(i)+1) + 1;
end

%%
function [rc_final rc_min_final rc_max_final] = find_candidate(v, BETA, ...
    K_GAMA, MIN_SIZE, MAX_SIZE)

length_v = length(v);

if(v(1) < BETA), where = 'b';
else where = 't';
end

beginnings = [];
ends = [];
for i = 2 : length_v,
    if(v(i) < BETA),
        if(where == 't'),
            beginnings = [beginnings 0];
            ends = [ends i];
            where = 'b';
        end
    else
        if(where == 'b'),
            beginnings = [beginnings i];
            ends = [ends 0];
            where = 't';
        end
    end
end
if(~length(beginnings)),
    beginnings = [1 0];
    ends = [0 length_v];
else
    if(~beginnings(1)),
        beginnings = [1 beginnings];
        ends = [0 ends];
    end
    if(~ends(end)),
        beginnings = [beginnings 0];
        ends = [ends length_v];
    end
end

rc = zeros([length(beginnings)/2 1]);
for i = 1 : 2 : length(beginnings),
    rc((i+1)/2) = find(v == max(v(beginnings(i) : ends(i+1))), 1, 'first');
end

rc_min = rc;
rc_max = rc;
for k = 1 : length(rc),
    for i = rc(k)-1 : -1 : 1,
        if(v(i) < K_GAMA*v(rc(k))),
            rc_min(k) = i;
            break;
        end
    end
    for i = rc(k)+1 : length_v,
        if(v(i) < K_GAMA*v(rc(k))),
            rc_max(k) = i;
            break;
        end
    end
end

rc_final = [];
rc_min_final = [];
rc_max_final = [];
for i = 1 : length(rc),
    len = rc_max(i) - rc_min(i) + 1;
    if(len >= MIN_SIZE && len <= MAX_SIZE),
        rc_final = [rc_final rc(i)];
        rc_min_final = [rc_min_final rc_min(i)];
        rc_max_final = [rc_max_final rc_max(i)];
    end
end
%figure(2); imshow(nimg);
