%% bayesclass.m
% From A First Course in Machine Learning, Chapter 5.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Modified by Shreya Garge 
% Naive Bayesian classifier using MLE
clear all;close all;

%% Load the data
load cbt1data

tmp=[1,2] 
nd=length(diseased)
nh=length(healthy)
t=[repmat(tmp(1),nd,1);repmat(tmp(2),nh,1)] %array with class labels
X=vertcat(diseased,healthy)                 %preparing the data
% Plot the data

cl = unique(t);                             %class labels
col = {'ko','kd','ks'}
fcol = {[1 0 0],[0 1 0],[0 0 1]};
figure(1);
hold off
for c = 1:length(cl)
    pos = find(t==cl(c));
    plot(X(pos,1),X(pos,2),col{c},...
        'markersize',10,'linewidth',2,...
        'markerfacecolor',fcol{c});
    hold on
end
xlim([-2 12])
ylim([-2 12])



%% Fit class-conditional Gaussians for each class
% Using the Naive assumption. i.e, not considering covariance since classes
% are assumed independent
for c = 1:length(cl)
    pos = find(t==cl(c));
    % Find the means
    class_mean(c,:) = mean(X(pos,:));
    class_var(c,:) = var(X(pos,:),1);
end


%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-2:0.1:12,-2:0.1:12);

Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    tempc = diag(class_var(c,:));
    const = -log(2*pi) - log(det(tempc));
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv));;
end

Probs = Probs./repmat(sum(Probs,3),[1,1,2]);

%% Plot the predictive contours
figure(1);hold off
for i = 1:2
    subplot(1,2,i);
    hold off
    for c = 1:length(cl)
        pos = find(t==cl(c));
        plot(X(pos,1),X(pos,2),col{c},...
            'markersize',10,'linewidth',2,...
            'markerfacecolor',fcol{c});
        hold on
    end
    xlim([-2 12])
    ylim([-2 12])
    
    contour(Xv,Yv,Probs(:,:,i));
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end


