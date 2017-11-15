% Computer Based Test 1 : 2 fold Cross Validation for Baysian
% Classification
% using 1-0 loss function
% Shreya Garge
clear all;
clc;
load('cbt1data.mat');  % Data contains information for healthy and diseased
                       % people as separate variables with two attributes
                       % each.
           
diseasedtest = diseased((end/2)+1:end,:);
healthytest = healthy((end/2)+1:end,:);
                                            %divided the training data into
                                            %two parts. one for training
                                            %and the other for testing
diseased = diseased(1:end/2,:);
healthy = healthy(1:end/2,:);

class = {diseased, healthy}; 
cn = 2;                % No. of classes and samples
total = 400;
%combining the diseased and healthy samples in order to use for testing
testdata = vertcat(diseasedtest,healthytest)'; 
%reference labels for 
%testing

testclasses = [repmat(1,length(diseasedtest),1) ; repmat(2,length(healthytest),1)]'; 

for i = 1:cn
    % Without Naive Bayes Assumption
    mean_class(:,i) = mean(class{i}', 2); % Calculating mean of attribute 1
                                         % and  2 of each class
    cov_class(:,:,i) = cov(class{i},1); % Calculating covariance between
                                         % attribute 1 and 2 for each class
                                         
    % With Naive Bayes Assumption 
    % Only the variance is considered as the variables are considered to be 
    % independent of each other given the class.
    var_class(:,i) = var(class{i},1)';
end

% Classification of new points 

% MLE - without Naive Bayes Assumption
for i = 1:cn
   sigmac = cov_class(:,:,i);           % Covariance matrix
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(testdata,1)* det(sigmac)); % Constant term
   for j = 1:size(testdata,2)
       x_u = testdata(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       mle(j,i) = const*exp(-power);    % Class conditional likelihood
   end
   
   [Xv,Yv] = meshgrid(0:0.2:10, 0:0.5:20);
   temp = [Xv(:)- mean_class(1,i) Yv(:)-mean_class(2,i)];
   const = -log(2*pi) - log(det(sigmac));
   Probs(:,:,i) = reshape(exp(const - 0.5*diag(temp*inv(sigmac)*temp')),size(Xv));
end

% cv loss for MLE without Naive 
[~, newptsclass] = max(mle,[],2); % Assign label as per highest probability
%for every point whose class is predicted wrongly
%increment count of loss
loss = 0;
for ct = 1:length(testclasses)
    if(testclasses(ct) ~= newptsclass(ct))  
        loss = loss+1;                     
    end
end
mleloss = loss/total;


% MLE - with Naive Bayes Assumption
for i = 1:cn
   sigmac = diag(var_class(:,i));       % Covariance matrix
                                        % diagonal as variance.
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(testdata,1)* det(sigmac)); % Constant term
   for j = 1:size(testdata,2)
       x_u = testdata(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       mle_n(j,i) = const*exp(-power);  % Class conditional likelihood
   end
   
   
end

% cv loss for MLE with Naive 
[~, newptsclass] = max(mle_n,[],2); % Assign label to highest probability
loss = 0;
for ct = 1:length(testclasses)
    if(testclasses(ct) ~= newptsclass(ct))
        loss = loss+1;
    end
end
mlenvloss = loss/total;

% MAP - without Naive Bayes Assumption
for i = 1:cn
   pc = size(class{i}',2)/total;         % Calculating probability of 
                                        % required class.
   sigmac = cov_class(:,:,i);           % Covariance matrix
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(testdata,1)* det(sigmac)); % Constant term
   for j = 1:size(testdata,2)
       x_u = testdata(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       map(j,i) = const*exp(-power)*pc; % Class conditional likelihood
                                        % * Probability of the class
   end
   
end
% cv loss for MAP without Naive 
[~, newptsclass] = max(map,[],2); % Assign label as per highest posterior
loss = 0;
for ct = 1:length(testclasses)
    if(testclasses(ct) ~= newptsclass(ct))
        loss = loss+1;
    end
end
maploss = loss/total;       


% MAP - with Naive Bayes Assumption
for i = 1:cn
   pc = size(class{i}',2)/total;         % Calculating priors
   sigmac = diag(var_class(:,i));       % Covariance matrix 
                                        % diagonal as variance.
   uc = mean_class(:,i);                % Mean vector
   const = 1/sqrt((2*pi)^size(testdata,1)* det(sigmac)); % Constant term
   for j = 1:size(testdata,2)
       x_u = testdata(:,j) - uc;          % Difference of new point vector
                                        % and mean vector
       power = 0.5*(x_u'*inv(sigmac)*x_u);
       map_n(j,i) = const*exp(-power)*pc; % Class conditional likelihood
                                        % * prior Probability of the class
   end
   
end

% cv loss for MAP with Naive 
[~, newptsclass] = max(map_n,[],2); % Assign label to highest posterior
loss = 0;
for ct = 1:length(testclasses)
    if(testclasses(ct) ~= newptsclass(ct))
        loss = loss+1;
    end
end
mapnvloss = loss/total; 


%%plot the respective losses for the four classifiers.

figure;
c = categorical({'MLE non naive','MLE naive','MAP non naive','MAP naive'});
losses = [mleloss mlenvloss maploss mapnvloss];
bar(c,losses);
hold on;
