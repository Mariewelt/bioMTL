function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units 
% hiddenSize: the number of hidden units  
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units.
% beta: weight of sparsity penalty term
% data: Matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

m = size(data,2);
a2 = [ ];
cost = 0;

     x = data;
     z2 = W1*x + repmat(b1,1,m);
     a2 = sigmoid(z2);
     rho = sum(a2,2)/m;
     z3 = W2*a2 + repmat(b2,1,m);
     a3 = sigmoid(z3);
     cost = ((a3-x).*(a3-x))/2;
     cost = sum(sum(cost));
     g = ones(size(a3));
     delta3 = -(x-a3).*(a3.*(g-a3));
     g = ones(size(a2));
     delta2 = (W2'*delta3 + repmat(beta*(-sparsityParam./rho + (1 - sparsityParam)./(1 - rho)),1,m)).*(a2.*(g-a2));
     W2grad = (delta3*a2')/m;
     W1grad = (delta2*x')/m;
     b2grad = sum(delta3,2)/m;
     b1grad = sum(delta2,2)/m;

W2grad = W2grad + lambda * W2;
W1grad = W1grad + lambda * W1;
fine = norm(W1,'fro')^2+norm(W2,'fro')^2;
sparFine = sparsityParam * log(sparsityParam./rho) + (1 - sparsityParam)*log((1 - sparsityParam)./(1 - rho));
cost = cost/m + (lambda*fine)/2 + beta * norm(sparFine,1);


grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end



function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

