function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Find C and sigma
vTest = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
R = zeros(64, 3);
i = 1;
for j = 1 : length(vTest)
	for k = 1 : length(vTest)
		R(i, 1) = vTest (1, j);
		R(i, 2) = vTest (1, k);
		model = svmTrain(X, y, R(i, 1), @(x1, x2) gaussianKernel(x1, x2, R(i, 2)));
		pred = svmPredict(model, Xval);
		R(i, 3) = mean(double(pred ~= yval));
		i ++;
	end
end
%i
%R 
[x, ix] = min(R(:, 3));
%R(ix, :)
C = R(ix, 1);
sigma = R(ix, 2);
%model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%pred = svmPredict(model, Xval);
%fprintf('mean=');
%mean(double(pred ~= yval))

% =========================================================================

end
