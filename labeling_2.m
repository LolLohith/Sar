function batchProcessSARImages(inputFolder)
    % ---- Setup Output Folders ----
    denoisedFolder = fullfile(inputFolder, 'Denoised');
    edgeFolder = fullfile(inputFolder, 'Cleaned_Edges');
    gaborFolder = fullfile(inputFolder, 'Gabor_Labelled');
    if ~exist(denoisedFolder, 'dir'), mkdir(denoisedFolder); end
    if ~exist(edgeFolder, 'dir'), mkdir(edgeFolder); end
    if ~exist(gaborFolder, 'dir'), mkdir(gaborFolder); end

    % ---- Supported Image Types ----
    imgExts = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};
    imageFiles = [];
    for i = 1:length(imgExts)
        imageFiles = [imageFiles; dir(fullfile(inputFolder, imgExts{i}))]; %#ok<AGROW>
    end

    % ---- First Pass: Extract Features Globally ----
    allFeatures = [];
    featureMaps = cell(1, length(imageFiles));
    imageSizes = zeros(length(imageFiles), 2);
    fprintf('Extracting Gabor features from all images...\n');

    for k = 1:length(imageFiles)
        filePath = fullfile(inputFolder, imageFiles(k).name);
        img = imread(filePath);
        if size(img, 3) == 3
            gimg = rgb2gray(img);
        else
            gimg = img;
        end

        I = double(gimg);
        I = I - min(I(:));
        I = I / max(I(:));  % Normalize to [0, 1]

        % Denoising
        sd = 0.25;
        j1 = imguidedfilter(I, 'NeighborhoodSize', [3, 3], 'DegreeOfSmoothing', sd);
        j2 = wdencmp('gbl', I, 'db5', 5, 0.5, 's', 1);
        processed = uint8((0.5 * j1 + 0.5 * j2) * 255);

        % Gabor Filtering
        I_norm = double(processed) / 255;
        gaborArray = gabor([4 8], [0 45 90 135]);
        gaborMag = imgaborfilt(I_norm, gaborArray);

        [nRows, nCols] = size(I_norm);
        nFilters = length(gaborArray);
        featureSet = zeros(nRows, nCols, nFilters);
        for i = 1:nFilters
            gaborMag(:,:,i) = imgaussfilt(gaborMag(:,:,i), 2);
            featureSet(:,:,i) = gaborMag(:,:,i);
        end

        X = reshape(featureSet, [], nFilters);
        X = (X - mean(X)) ./ std(X);

        featureMaps{k} = X;
        allFeatures = [allFeatures; X]; %#ok<AGROW>
        imageSizes(k,:) = [nRows, nCols];
    end

    % ---- Global KMeans Clustering ----
    numClusters = 4;
    fprintf('Performing global KMeans clustering...\n');
    [~, C] = kmeans(allFeatures, numClusters, 'MaxIter', 500, 'Replicates', 3);

    % ---- Second Pass: Assign Clusters & Save Outputs ----
    fprintf('Applying cluster labels and saving results...\n');
    fixedColors = [255 0 0; 0 255 0; 0 0 255; 255 255 0] / 255;

    for k = 1:length(imageFiles)
        file = imageFiles(k);
        filePath = fullfile(inputFolder, file.name);
        [~, name, ~] = fileparts(file.name);

        % Reload image
        img = imread(filePath);
        if size(img, 3) == 3
            gimg = rgb2gray(img);
        else
            gimg = img;
        end

        I = double(gimg);
        I = I - min(I(:));
        I = I / max(I(:));  % Normalize to [0, 1]

        % Denoising
        sd = 0.25;
        j1 = imguidedfilter(I, 'NeighborhoodSize', [3, 3], 'DegreeOfSmoothing', sd);
        j2 = wdencmp('gbl', I, 'db5', 5, 0.5, 's', 1);
        processed = uint8((0.5 * j1 + 0.5 * j2) * 255);

        % Edge Detection & Cleaning
        edges = edge(processed, 'Canny');
        edges_cleaned = imclose(edges, strel('disk', 2));
        edges_cleaned = imfill(edges_cleaned, 'holes');
        edges_cleaned = bwareaopen(edges_cleaned, 50);

        % Assign cluster using precomputed centroids
        X = featureMaps{k};
        idx = knnsearch(C, X);
        labelled_gabor = reshape(idx, imageSizes(k,1), imageSizes(k,2));
        rgbLabelled = label2rgb(labelled_gabor, fixedColors);

        % Save outputs
        imwrite(processed, fullfile(denoisedFolder, [name '_denoised.png']));
        imwrite(edges_cleaned, fullfile(edgeFolder, [name '_edge.png']));
        imwrite(rgbLabelled, fullfile(gaborFolder, [name '_gabor_labelled.png']));
    end

    disp('Batch processing complete.');
end
