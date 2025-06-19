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

    % ---- Loop Over Images ----
    for k = 1:length(imageFiles)
        file = imageFiles(k);
        filePath = fullfile(inputFolder, file.name);
        [~, name, ~] = fileparts(file.name);
        fprintf('Processing: %s\n', file.name);

        % ---- Step 1: Load and preprocess image ----
        img = imread(filePath);
        if size(img, 3) == 3
            gimg = rgb2gray(img);
        else
            gimg = img;
        end

        I = double(gimg);
        I = I - min(I(:));
        I = I / max(I(:));  % Normalize to [0, 1]

        % ---- Step 2: Denoising ----
        sd = 0.05;
        j1 = imguidedfilter(I, 'NeighborhoodSize', [3, 3], 'DegreeOfSmoothing', sd);
        j2 = wdencmp('gbl', I, 'db5', 5, 0.5, 's', 1);
        processed = uint8((0.5 * j1 + 0.5 * j2) * 255);

        % ---- Step 3: Edge Detection ----
        edges = edge(processed, 'Canny');

        % ---- Step 4: Morphological Cleaning ----
        edges_cleaned = imclose(edges, strel('disk', 2));
        edges_cleaned = imfill(edges_cleaned, 'holes');
        edges_cleaned = bwareaopen(edges_cleaned, 50);

        % ---- Step 5: Gabor-Based Labeling ----
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
        numClusters = 4;
        cluster_idx = kmeans(X, numClusters, 'MaxIter', 500, 'Replicates', 3);
        labelled_gabor = reshape(cluster_idx, nRows, nCols);

        % ---- Step 6: Save Results ----
        imwrite(processed, fullfile(denoisedFolder, [name '_denoised.png']));
        imwrite(edges_cleaned, fullfile(edgeFolder, [name '_edge.png']));
        imwrite(label2rgb(labelled_gabor), fullfile(gaborFolder, [name '_gabor_labelled.png']));
    end

    disp('Batch processing complete.');
end
