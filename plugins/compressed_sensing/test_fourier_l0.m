%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run Fourier L0 compression on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = 'D:/projects/compareFramework/plugins/cfft/dataset/';
IMAGE_NAME = 'lenna.tif/';
IMAGE_SIZE = [500, 500];
BLOCK_SIZE = 50;
RATIO = [0.2, 0.05, 0.01];
OVERLAP_PERCENT = 0.5;

% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

figure;
subplot(2, 2, 1);
imshow(img, []);
title('Original Image');

for i = 1:3
   k = RATIO(i) * BLOCK_SIZE * BLOCK_SIZE;
   blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);

   [M, N, B] = size(blocks);
   [fourier_basis, block_coefficients] = compressFourierL0(blocks, k);
   reconstructed_blocks = reconstructBlocks(fourier_basis, block_coefficients, ...
                                            M, N);
   reconstruction = assembleBlocks(real(reconstructed_blocks), BLOCK_SIZE, ...
                                   IMAGE_SIZE, OVERLAP_PERCENT);

    % Display.
    subplot(2, 2, i+1);
    imshow(reconstruction, []);
    title(sprintf('Compression ratio: %2.1f%%', 100 * RATIO(i)));
end
