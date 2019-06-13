
function[a p_index]=accuracy(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X,Y)

 %reshaping
  
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params(1+(hidden_layer2_size*(hidden_layer1_size+1))+((hidden_layer1_size) * (input_layer_size + 1)):end), ...
                 num_labels, (hidden_layer2_size + 1));
 %useful variable
    m=size(X,1);
    theta1_grad=zeros(size(Theta1));
    theta2_grad=zeros(size(Theta2));
    theta3_grad=zeros(size(Theta3));
    
    %recode y to Y
    
    I=eye(num_labels);
    y=zeros(m,num_labels);
    
    for i=1:m
      
      y(i,:)=I(Y(i),:);
      
      end  
    
    %forward propogation
    
    a1=[ones(m,1)  X];                    %input layer activation
    z2=a1*Theta1';         
    a2=[ones(size(z2, 1), 1)  sigmoid(z2)];      %hidden layer activation
    z3=a2*Theta2';
    a3=[ones(size(z3,1),1) sigmoid(z3)];
    z4=a3*Theta3';
    a4=sigmoid(z4);
    h=a4;
    
    
    
    [p_values,p_index]=max(h,[],2); 
    p=p_index;
    %accuracy
    a=mean(double(p==Y))*100;
    end
  
      
      
      