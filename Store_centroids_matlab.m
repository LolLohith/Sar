% Sample data
X = rand(100, 3);
numClusters = 4;

% K-means
[idx, centroids] = kmeans(X, numClusters);
colors = lines(numClusters);

% Define your custom save path
saveFolder = 'C:/Users/YourName/Documents/ClusteringResults';
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);  % Create folder if it doesn't exist
end

% Full file path
savePath = fullfile(saveFolder, 'kmeans_model.mat');

% Save centroids and colors
save(savePath, 'centroids', 'colors');
