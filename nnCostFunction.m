
function [J,grad]=nnCostFunction(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X_train,Y_train)
  
  %reshaping
  
 Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params(1+(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1)):end), ...
                 num_labels, (hidden_layer2_size + 1));
                 
 %useful variable
    m=size(X_train,1);
    theta1_grad=zeros(size(Theta1));
    theta2_grad=zeros(size(Theta2));
    theta3_grad=zeros(size(Theta3));
    J=0;    
    lambda=0;
    %recode y to Y
    
    I=eye(num_labels);
    y=zeros(m,num_labels);
    
    for i=1:m
      
      y(i,:)=I(Y_train(i),:);
      
      end  
    
    %forward propogation
    
    a1=[ones(m,1)  X_train];                    %input layer activation
    z2=a1*Theta1';         
    a2=[ones(size(z2, 1), 1)  sigmoid(z2)];      %hidden layer activation
    z3=a2*Theta2';
    a3=[ones(size(z3,1),1) sigmoid(z3)];
    z4=a3*Theta3';
    a4=sigmoid(z4);
    h=a4; 
    
    %cost function
    penalty=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2,2))+sum(sum(Theta2(:,2:end).^2,2))+sum(sum(Theta3(:,2:end).^2,2)));
    J=sum(sum(-y.*log(h)-(1-y).*log(1-h),2))/(2*m);
     J=J+penalty;    
    %backpropogation
    del4=h.-y;   %error in output layer
    del3 = (del4*Theta3).*sigmoidGradient([ones(size(z3, 1), 1) z3]); %error in hidden layer2
    del3 = del3(:, 2:end);
    del2 = (del3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]); %error in hidden layer2
    del2 = del2(:, 2:end);
    
    delta_1 = (del2'*a1);
    delta_2 = (del3'*a2);
    delta_3 = (del4'*a3);
     
    theta1_grad=delta_1./m+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
    theta2_grad=delta_2./m+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
    theta3_grad=delta_3./m+(lambda/m)*[zeros(size(Theta3,1),1) Theta3(:,2:end)];
    % Unroll gradients
    grad = [theta1_grad(:) ; theta2_grad(:) ; theta3_grad(:)]; 
       
      end
    
    
    
    
    
    