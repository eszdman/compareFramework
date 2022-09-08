%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script. Run reversed Huber compressed sensing on an image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters.
IMAGE_PATH = '../../data/';
IMAGE_NAME = 'lenna.png';

IMAGE_SIZE = [512, 512];
BLOCK_SIZE = 8;
rho = [0.01 0.1 1.0 10];
ALPHA = 1.0;
OVERLAP_PERCENT = 0;
BASIS_OVERSAMPLING = 0.1:0.2:1.5;


% Import the image.
img = imresize(rgb2gray(imread([IMAGE_PATH, IMAGE_NAME])), IMAGE_SIZE);

for i = 1:length(rho)
    for j = 1:numel(BASIS_OVERSAMPLING)
        basis_oversampling = BASIS_OVERSAMPLING(j);
        alpha = ALPHA;
        RHO = rho(i)
        fprintf('RHO = %1.1f, OSR = %1.1f\n', RHO, basis_oversampling);
        
        blocks = getBlocks(img, BLOCK_SIZE, OVERLAP_PERCENT);
        
        [M, N, B] = size(blocks);
        [dct_basis, block_coefficients] = compressedSenseDCTHuber(blocks, RHO, alpha, ...
            basis_oversampling);
        reconstructed_blocks = reconstructBlocks(dct_basis, block_coefficients, ...
            M, N);
        reconstruction = assembleBlocks(reconstructed_blocks, BLOCK_SIZE, ...
            IMAGE_SIZE, OVERLAP_PERCENT);
        
        
     
        % Save coefficients to file.
        filename = sprintf('../../reconstructions/matlab figures/cs_dct_huber_size%dx%d_rho%1dp%1d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.mat', ...
            IMAGE_SIZE(1), IMAGE_SIZE(2), floor(RHO), 10*mod(RHO, 1), floor(alpha), 10*mod(alpha, 1), ...
            floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
            floor(basis_oversampling), 10*mod(basis_oversampling, 1));
        save(filename, 'block_coefficients');
        
        
        % Display.
%         figure;
%         imshow(reconstruction, []);
        %title(sprintf('Alpha: %f', alpha));
        
        filename = sprintf('../../reconstructions/matlab figures/cs_dct_huber_size%dx%d_rho%1dp%1d_alpha%1dp%1d_overlap%1dp%1d_oversample%1dp%1d.png', ...
            IMAGE_SIZE(1), IMAGE_SIZE(2), floor(RHO), 10*mod(RHO, 1), floor(alpha), 10*mod(alpha, 1), ...
            floor(OVERLAP_PERCENT), 10*mod(OVERLAP_PERCENT, 1), ...
            floor(basis_oversampling), 10*mod(basis_oversampling, 1));
        scaled_reconstruction = reconstruction / max(max(reconstruction));
        imwrite(scaled_reconstruction, filename);
    end
end
