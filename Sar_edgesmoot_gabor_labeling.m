% Clear environment
clear; clc; close all;

% ---- Step 1: Load and preprocess SAR image ----
image_path = 'single-polarization-radar-image-1024x492.jpg'; % <-- Change this if needed
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
I = I / max(I(:));  % Normalize to [0, 1]

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
edges = edge(processed, 'Canny');
figure, imshow(edges), title('Edge Map');

% ---- Step 4: Morphological Cleaning ----
edges_cleaned = imclose(edges, strel('disk', 2));
edges_cleaned = imfill(edges_cleaned, 'holes');
edges_cleaned = bwareaopen(edges_cleaned, 50);
figure, imshow(edges_cleaned), title('Cleaned Edges');

% ---- Step 5: Gabor-Based Labeling ----
% Convert processed image to double in range [0,1] for Gabor filtering
I_norm = double(processed) / 255;

% Define Gabor filter bank (adjust orientations/frequencies as needed)
gaborArray = gabor([4 8], [0 45 90 135]);  % 2 scales, 4 orientations

% Apply Gabor filter bank
gaborMag = imgaborfilt(I_norm, gaborArray);

% Create feature vector for each pixel
[nRows, nCols] = size(I_norm);
nFilters = length(gaborArray);
featureSet = zeros(nRows, nCols, nFilters);

for i = 1:nFilters
    % Optionally smooth each Gabor response
    gaborMag(:,:,i) = imgaussfilt(gaborMag(:,:,i), 2);
    featureSet(:,:,i) = gaborMag(:,:,i);
end

% Reshape to 2D feature matrix for clustering
X = reshape(featureSet, [], nFilters);
X = (X - mean(X)) ./ std(X);  % Z-score normalization

% Apply k-means clustering
numClusters = 4;  % Adjust this for more/fewer regions
cluster_idx = kmeans(X, numClusters, 'MaxIter', 500, 'Replicates', 3);

% Reshape to labeled image
labelled_gabor = reshape(cluster_idx, nRows, nCols);

% Display Gabor-based labeling
figure, imshow(label2rgb(labelled_gabor)), title('Gabor-Based Labeled Regions');

% ---- Step 6: Optional - Save Results ----
imwrite(processed, 'denoised_image.png');
imwrite(edges_cleaned, 'edges.png');
imwrite(label2rgb(labelled_gabor), 'gabor_labelled.png');
