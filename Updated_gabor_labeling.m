function processImagesWithGaborAndDenoise(inputFolder)
    % Create output directories
    edgeFolder = fullfile(inputFolder, 'Cleaned_Edges');
    gaborFolder = fullfile(inputFolder, 'Gabor_Images');
    
    if ~exist(edgeFolder, 'dir'), mkdir(edgeFolder); end
    if ~exist(gaborFolder, 'dir'), mkdir(gaborFolder); end

    % Supported image extensions
    imgExts = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
    imageFiles = [];

    % Collect all image files
    for i = 1:length(imgExts)
        imageFiles = [imageFiles; dir(fullfile(inputFolder, imgExts{i}))]; %#ok<AGROW>
    end

    if isempty(imageFiles)
        disp('No image files found.');
        return;
    end

    % Define Gabor filter bank
    gaborArray = gabor([4 8], [0 45 90 135]);

    % Process each image
    for k = 1:length(imageFiles)
        fileName = imageFiles(k).name;
        filePath = fullfile(inputFolder, fileName);
        [~, name, ~] = fileparts(fileName);

        % Read image
        img = imread(filePath);
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        % ------------ Step 1: Denoising ------------
        denoisedImg = imnlmfilt(grayImg);  % Non-local means filter

        % ------------ Step 2: Edge Detection & Cleaning ------------
        edges = edge(denoisedImg, 'Canny');
        edgesClean = imclose(edges, strel('disk', 1));
        edgesClean = imfill(edgesClean, 'holes');
        edgesClean = bwareaopen(edgesClean, 30);

        % Save cleaned edge image
        edgeOutPath = fullfile(edgeFolder, [name '_edge.png']);
        imwrite(edgesClean, edgeOutPath);

        % ------------ Step 3: Gabor Filtering ------------
        gaborMag = imgaborfilt(denoisedImg, gaborArray);

        % Combine Gabor magnitude responses
        gaborSum = sum(gaborMag, 3);
        gaborNorm = mat2gray(gaborSum);  % Normalize to [0 1]

        % Save Gabor result
        gaborOutPath = fullfile(gaborFolder, [name '_gabor.png']);
        imwrite(gaborNorm, gaborOutPath);

        fprintf('Processed: %s\n', fileName);
    end

    disp('All images have been processed.');
end
