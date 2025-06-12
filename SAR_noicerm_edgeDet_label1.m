% Clear environment
clear; clc; close all;

% ---- Step 1: Load and preprocess SAR image ----
image_path = 'single-polarization-radar-image-1024x492.jpg'; % <-- Change this to your actual image path
img = imread(image_path);

% Convert to grayscale if RGB
if size(img, 3) == 3
    gimg = rgb2gray(img);
else
    gimg = img;
end

% Normalize and convert to double
I = double(gimg);
I = I - min(I(:));
if max(I(:)) > 1
    I = I / 255;
end

% ---- Step 2: Denoising ----
sd = 0.05;

% Guided filter
j1 = imguidedfilter(I, 'NeighborhoodSize', [3, 3], 'DegreeOfSmoothing', sd);

% Wavelet denoising
j2 = wdencmp('gbl', I, 'db5', 5, 0.5, 's', 1);  % global thresholding, db5 wavelet

% Combine filtered images
processed = uint8((0.5 * j1 + 0.5 * j2) * 255);

% Show denoised image
figure, imshow(processed), title('Denoised SAR Image');

% ---- Step 3: Edge Detection ----
edges = edge(processed, 'Canny');  % use 'Sobel' or 'Prewitt' as alternatives
figure, imshow(edges), title('Edge Map');

% ---- Step 4: Morphological Cleaning ----
edges_cleaned = imclose(edges, strel('disk', 2));  % close small gaps
edges_cleaned = imfill(edges_cleaned, 'holes');    % fill enclosed regions

% Optional: remove small objects
edges_cleaned = bwareaopen(edges_cleaned, 50);     % remove small noise

% Show cleaned edge map
figure, imshow(edges_cleaned), title('Cleaned Edges');

% ---- Step 5: Labelling the Segmented Regions ----
[labelled, num_regions] = bwlabel(edges_cleaned, 4);
figure, imshow(label2rgb(labelled)), title(['Labelled Regions: ', num2str(num_regions)]);

% ---- Step 6: Optional - Save Results ----
imwrite(processed, 'denoised_image.png');
imwrite(edges_cleaned, 'edges.png');
