function nimg = suryanarayana(img, MIN_HEIGHT_CHAR, MAX_HEIGHT_CHAR, ...
    MIN_WIDTH_VLP, MAX_DISTANCE_CHARS, FINAL_DILATION_SIZE)

if(ischar(img)), img = imread(img);
end
%figure(1); imshow(img);

if(nargin == 1),
    MIN_HEIGHT_CHAR = 12;
    MAX_HEIGHT_CHAR = 53;
    MIN_WIDTH_VLP = 67;
    MAX_DISTANCE_CHARS = 31;
    FINAL_DILATION_SIZE = 11;
end

if(ndims(img) == 3), img = rgb2gray(img);
end
img = im2double(img);

[height width] = size(img);

% Vertical edge detection
img_proc = abs(filter2(fspecial('sobel')' , img));

% Binarization
img_proc = img_proc >= graythresh(img_proc); % Otsu

% To ensure that the license plate is not cropped...
se = ones([1 MAX_DISTANCE_CHARS]);
img_proc = imclose(img_proc, se);

% Eliminate the regions whose height is less than the minimum character
% height
se = ones([MIN_HEIGHT_CHAR 1]);
img_proc = imopen(img_proc, se);

% Eliminate regions with height greater that the maximum character height
se = ones([MAX_HEIGHT_CHAR 1]);
img_with_high_columns = imopen(img_proc, se);
img_proc = img_proc - img_with_high_columns;

% Eliminate noise blobs whose width is less than the minimum width of
% license plate
se = ones([1 MIN_WIDTH_VLP]);
img_proc = imopen(img_proc, se);

% Final adjustment based on the database features
[l num] = bwlabel(img_proc, 8);
nimg = zeros([height width]);

for i = 1 : num,
    [rows cols] = find(l == i);
    
    img_candidate = zeros([height width]);
    img_candidate(min(rows) : max(rows) , min(cols) : max(cols)) = 1;

    se = ones([FINAL_DILATION_SIZE 1]);
    img_candidate = imdilate(img_candidate, se);

    nimg = nimg + img_candidate;
end
%figure(2); imshow(nimg);
