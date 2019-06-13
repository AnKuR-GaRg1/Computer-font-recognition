function g = sigmoidGradient(z)
    g = zeros(size(z));
    g1= 1.0 ./ (1.0 + exp(-z));
    g=g1.*(1-g1);

end
