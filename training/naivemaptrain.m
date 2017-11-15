% Computer Based Test 1 : Bayesian Classification
% Classifying Unseen data with naive bayesian calculator using MAP
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

% Training


for i = 1:cn
    
    mean_class(:,i) = mean(class{i}', 2); % Calculating mean of attribute 1
                                         % and  2 of each class
    %cov_class(:,:,i) = cov(class{i},1); % Calculating covariance between
                                         % attribute 1 and 2 for each class
                                         
    % With Naive Bayes Assumption 
    %covariance is not required as the variables are considered to be 
    % independent of each other given the class.
    var_class(:,i) = var(class{i},1)';
end
unseen=unseen';



% MAP - with Naive Bayes Assumption
for i = 1:cn
   pc = size(class{i}',2)/total;         % Calculating prior of 
                                        % required class.
   sigmac = diag(var_class(:,i));       % Covariance matrix with 
                                        % diagonal as variance.
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(unseen,1)* det(sigmac)); % Constant term
   for j = 1:size(unseen,2)
       x_u = unseen(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       map_n(j,i) = const*exp(-power)*pc; % Class conditional likelihood
                                        % * prior Probability of the class
   end
   
end
nod=0;
noh=0;
% Plotting the figures for MAP with Naive assumption
[~, newptsclass] = max(map_n,[],2); % Assign label to highest posterior
for sh = 1:length(newptsclass)
    if(newptsclass(sh)==1)
        nod=nod+1;
    else
        noh=noh+1;
    end
end
col_train = {'ro','bo'};
col_test = {'rx','bx'};
figure;


for c = 1:cn
    pos_new = find(newptsclass == c);
    map = size(pos_new)
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

title('MAP With Naive')
xlabel('Attribute 1')
ylabel('Attribute 2')
