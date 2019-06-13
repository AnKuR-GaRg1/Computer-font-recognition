
% computer font regognition project


%loading train dataset
dataset_train=csvread("mnist_train.csv");
%loading test dataset

dataset_test=csvread("mnist_test.csv");

%useful parmeters

input_layer_size  = 784;    % 40x40 Input Images of Digits
hidden_layer1_size = 100;  
hidden_layer2_size = 100;
num_labels = 62;          % 62 labels, from 1 to 10 and ato z and A to Z    



%training data

m_train= size(dataset_train,1);
n_train=784;

X_train=dataset_train(:,2:end);
Y_train=dataset_train(:,1);

clear dataset_train;
%test set
m_test= size(dataset_test,1);
n_test=784;

X_test=dataset_test(:,2:n_test+1);
Y_test=dataset_test(:,1);

clear dataset_test;

%normalization
%
%Xmean=repmat(mean(X_train),m_train,1);
%Xstd=repmat(std(X_train),m_train,1);
%X_norm=(X_train-Xmean)./Xstd;
%X_train=X_norm;

%Xmean=repmat(mean(X_test),m_test,1);
%Xstd=repmat(std(X_test),m_test,1);
%X_norm=(X_test-Xmean)./Xstd;
%X_test=X_norm;

%clear X_norm;
%clear Xmean;
%clear Xstd;
%randomize weights
epsilon=0.2;
Theta1 = (rand(hidden_layer1_size,input_layer_size+1)*(2*epsilon))-epsilon;
Theta2 = (rand(hidden_layer2_size,hidden_layer1_size+1)*(2*epsilon))-epsilon;
Theta3 = (rand(num_labels,hidden_layer2_size+1)*(2*epsilon))-epsilon;


  
nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:)];

% cost function
 

          options=optimset('MaxIter',100);
          costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X_train,Y_train);    
          [grad, cost] = fmincg(costFunction,nn_params,options);     
          


%reshaping gradient matrix
Theta_grad1 = reshape(grad(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta_grad2 = reshape(grad((1 + (hidden_layer1_size *  (input_layer_size + 1))):(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta_grad3 = reshape(grad(1+(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1)):end), ...
                 num_labels, (hidden_layer2_size + 1));



   
  
    nn_params = [Theta_grad1(:) ; Theta_grad2(:); Theta_grad3(:)];
   [a_train p_index]=accuracy(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X_train,Y_train);
   [a_test p_index]=accuracy(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X_test,Y_test);
    
    
    

