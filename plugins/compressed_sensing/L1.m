function res = L1(imagepath, sizex, sizey, block, ratio)
% Parameters.

IMAGE_SIZE = [sizex, sizey];
BLOCK_SIZE = block;
ALPHA = ratio;
OVERLAP_PERCENT = 0;
BASIS_OVERSAMPLING = 0.1:0.1:1.5;

% Import the image.
img = imresize(imread(imagepath), IMAGE_SIZE);

basis_oversampling = BASIS_OVERSAMPLING(1);
blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);

[M, N, B] = size(blocks);
block_coefficients = compressedSenseImgL1(blocks, ratio, ...
                                                      basis_oversampling);
reconstructed_blocks = reconstructBlocks(eye(M * N), block_coefficients, ...
                                        M, N);
reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
                               IMAGE_SIZE, OVERLAP_PERCENT);
res = reconstruction / max(max(reconstruction));
end
