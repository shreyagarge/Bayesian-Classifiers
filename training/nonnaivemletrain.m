% Computer Based Test 1 : Bayesian Classification
% Classifying Unseen data with bayesian calculator using MLE without naive
% assumption
% Parts of code taken from Simon Rogers - A first course in Machine
% Learning
% Shreya Garge
clear all;
clc;
load('cbt1data.mat');  % Data contains information for healthy and diseased
                       % people as separate variables with two attributes
                       % each - chemical 1 and chemical 2.

class = {diseased, healthy}; 
cn = 2;               
total = 800;           % number of training samples

% Training the model



for i = 1:cn
    % Without Naive Bayes Assumption
    mean_class(:,i) = mean(class{i}', 2); % Calculating mean of attribute 1
                                         % and  2 of each class
    cov_class(:,:,i) = cov(class{i},1); % Calculating covariance between
                                         % attribute 1 and 2 for each class
                                         
    
    var_class(:,i) = var(class{i},1)';
end

% Classification of new points 
unseen = unseen';
% MLE - without Naive Bayes Assumption
for i = 1:cn
   sigmac = cov_class(:,:,i);           % Covariance matrix
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(unseen,1)* det(sigmac)); % Constant term
   for j = 1:size(unseen,2)
       x_u = unseen(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       mle(j,i) = const*exp(-power);    % Class conditional likelihood
   end
  
   [Xv,Yv] = meshgrid(0:0.2:10, 0:0.5:20);
   temp = [Xv(:)- mean_class(1,i) Yv(:)-mean_class(2,i)];
   const = -log(2*pi) - log(det(sigmac));
   Probs(:,:,i) = reshape(exp(const - 0.5*diag(temp*inv(sigmac)*temp')),size(Xv));
end
nod=0;
noh=0;
% Plotting the figures for MLE without Naive (From [1])
[~, newptsclass] = max(mle,[],2); % Assign label as per highest probability
for sh = 1:length(newptsclass)
    if(newptsclass(sh)==1)
        nod=nod+1;
    else                %calculating number of samples classified into each class
        noh=noh+1;
    end
end
col_train = {'ro','bo'};
col_test = {'rx','bx'};
figure;

for c = 1:cn
    pos_new = find(newptsclass == c);
    mle = size(pos_new)
    plot(class{c}(1,:),class{c}(2,:), col_train{c},'markersize',10, ...
         'linewidth',2);
    hold on;
    plot(unseen(1,pos_new),unseen(2, pos_new),col_test{c},'markersize', ...
        10,'linewidth',2);
    hold on;
end

xlim([-2 12])
ylim([-2 12])
xlabel('Attribute 1')
ylabel('Attribute 2')

legend('Diseased Training Sample', 'Diseased New Sample', ...
       'Healthy Training Sample', 'Healthy New Sample')
title('MLE Without Naive')





                                            