function res = CFFT(imagepath, sizex, sizey, block, ratio)
%#codegen
IMAGE_SIZE = [sizex, sizey];
BLOCK_SIZE = block;
RATIO = ratio;
OVERLAP_PERCENT = 0.5;
% Import the image.
img = imresize((imread(imagepath)), IMAGE_SIZE);
k = RATIO * BLOCK_SIZE * BLOCK_SIZE;
blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
[M, N] = size(blocks);
[fourier_basis, block_coefficients] = compressFourierL0(blocks, k);
reconstructed_blocks = reconstructBlocks(fourier_basis, block_coefficients, ...
                                        M, N);
reconstruction = assembleBlocks(real(reconstructed_blocks), BLOCK_SIZE, ...
                               IMAGE_SIZE, OVERLAP_PERCENT);
res = reconstruction;
end