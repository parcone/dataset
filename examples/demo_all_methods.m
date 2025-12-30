% Demo script to test all denoising methods
% This script demonstrates how to use each denoising method

clear; clc; close all;

% Add paths
addpath(genpath('../utils'));
addpath(genpath('../methods'));

% Load or create a test image
% For demo purposes, create a synthetic fluorescence image
test_image = phantom('Modified Shepp-Logan', 256);
test_image = test_image * 100; % Scale to fluorescence intensity range

% Add Poisson-Gaussian noise (simulating fluorescence microscopy)
sigma = 5; % Gaussian noise standard deviation
a = 0.1;  % Poisson scaling factor
noisy_image = a * poissrnd(test_image / a) + sigma * randn(size(test_image));

% Display original and noisy images
figure('Position', [100, 100, 1200, 800]);
subplot(3, 3, 1); imshow(test_image, []); title('Original Image');
subplot(3, 3, 2); imshow(noisy_image, []); title('Noisy Image');

% List of all methods
methods = {
    'VST_NLM', 'denoise_VST_NLM';
    'VST_BM3D', 'denoise_VST_BM3D';
    'VST_KSVD', 'denoise_VST_KSVD';
    'VST_KSVDD', 'denoise_VST_KSVDD';
    'VST_KSVDG', 'denoise_VST_KSVDG';
    'VST_EPLL', 'denoise_VST_EPLL';
    'VST_WNNM', 'denoise_VST_WNNM';
    'PURE_LET', 'denoise_PURE_LET'
};

% Test each method
results = cell(length(methods), 3);
for i = 1:length(methods)
    method_name = methods{i, 1};
    function_name = methods{i, 2};
    
    fprintf('Testing %s...\n', method_name);
    
    try
        tic;
        % Call the denoising function
        denoised = feval(function_name, noisy_image);
        processing_time = toc;
        
        results{i, 1} = method_name;
        results{i, 2} = denoised;
        results{i, 3} = processing_time;
        
        % Display result
        subplot(3, 3, i+2);
        imshow(denoised, []); 
        title(sprintf('%s (%.2fs)', method_name, processing_time));
        
        fprintf('  - Processing time: %.2f seconds\n', processing_time);
        
    catch ME
        fprintf('  - Error: %s\n', ME.message);
        results{i, 1} = method_name;
        results{i, 2} = [];
        results{i, 3} = NaN;
        
        subplot(3, 3, i+2);
        text(0.5, 0.5, 'Error', 'HorizontalAlignment', 'center');
        title(sprintf('%s (Error)', method_name));
    end
end

% Calculate PSNR values
fprintf('\nPSNR Results:\n');
fprintf('Method\t\tPSNR (dB)\tTime (s)\n');
fprintf('--------------------------------\n');

for i = 1:length(results)
    if ~isempty(results{i, 2})
        psnr_value = psnr(results{i, 2}, test_image);
        fprintf('%-12s\t%.2f\t\t%.2f\n', results{i, 1}, psnr_value, results{i, 3});
    else
        fprintf('%-12s\tError\t\t-\n', results{i, 1});
    end
end

% Save results
save('denoising_results.mat', 'results', 'test_image', 'noisy_image');
fprintf('\nResults saved to denoising_results.mat\n');
