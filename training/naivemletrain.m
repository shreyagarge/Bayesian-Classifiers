% Computer Based Test 1 : Bayesian Classification
% Classifying Unseen data using naive bayesian calculator using MLE
% Parts of code taken from Simon Rogers - A first course in Machine
% Learning
% Shreya Garge

clear all; close all;
clc;
load('cbt1data.mat');  % Data contains information for healthy and diseased
                       % people as separate variables with two attributes
                       % each.

class = {diseased, healthy}; 
cn = 2;                % No. of classes
total = 800;           % Total no. of training samples

% Training the model 
for i = 1:cn
    
    mean_class(:,i) = mean(class{i}', 2); % Calculating mean of attribute 1
                                         % and  2 of each class
    %cov_class(:,:,i) = cov(class{i},1); % Calculating covariance between
                                         % attribute 1 and 2 for each class
                                         
    % With Naive Bayes Assumption 
    % the covariance is not considered as the variables are considered to be 
    % independent of each other given the class.
    var_class(:,i) = var(class{i},1)';  %calculating variance of attribute 1
                                        %and 2 of each class
end
unseen=unseen';
% MLE - with Naive Bayes Assumption
for i = 1:cn
   sigmac = diag(var_class(:,i));       % Covariance matrix with 
                                        % diagonal as variance.
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(unseen,1)* det(sigmac)); % preparing to fit 
   for j = 1:size(unseen,2)                            %a gaussian
       x_u = unseen(:,j) - uc;          % Difference: unseen points vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       mle_n(j,i) = const*exp(-power);  % Class conditional likelihood
   end
   
   
end
nod=0;
noh=0;
% Plotting the figures for MLE with Naive assumption
[~, newptsclass] = max(mle_n,[],2); % Assign highest probability class

for sh = 1:length(newptsclass)
    if(newptsclass(sh)==1)
        nod=nod+1;          %calculting number of samples classified to 
    else                    %each class
        noh=noh+1;
    end
end
col_train = {'ro','bo'};
col_test = {'rx','bx'};
figure;
for c = 1:cn
    pos_new = find(newptsclass == c);
    mle_n = size(pos_new)
    plot(class{c}(1,:),class{c}(2,:), col_train{c},'markersize',10, ...
         'linewidth',2);
    hold on;
    plot(unseen(1,pos_new),unseen(2, pos_new),col_test{c},'markersize', ...
        10,'linewidth',2);
    hold on;
end

xlim([-2 12])
ylim([-2 12])
legend('Diseased Training Sample', 'Diseased New Sample', ...
       'Healthy Training Sample', 'Healthy New Sample')
title('MLE With Naive')
xlabel('Attribute 1')
ylabel('Attribute 2')
