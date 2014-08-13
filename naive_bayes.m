% Naive Bayes classifier assignment
data = load ('nursery_data.txt');

num_feat = 8;
feat_dims = [3 5 4 4 3 2 3 3];
num_classes = 5;

reset(RandStream.getDefaultStream);
% data = [data(1:end-1,:);repmat(data(end-1,:),2000,1)];
data = data(randperm(size(data,1)), :);
train = data(1:size(data, 1)/2, :);
test = data(size(data,1)/2+1:end,:);

% Training
prior_class = zeros(num_classes, 1);

for i = 1:size(train, 1) 
	prior_class(train(i, end)) = prior_class(train(i, end)) + 1;
end

prior_class = (prior_class + 1) / (size(train, 1) + num_classes);

train_prob = {};

for i = 1:num_feat
	feat_train_prob = zeros(feat_dims(i), num_classes);
	for j = 1 : feat_dims(i)
		for k = 1 : num_classes
			feat_count = length(find(train(:,i) == j & train(:,end) == k ));
			class_count = length(find(train(:,end) == k));
			feat_train_prob(j,k) = (feat_count + 1 ) / (class_count + num_feat);  % With Laplace Smoothning
		end
	end
	train_prob = [train_prob; {feat_train_prob}];
end


prior_class;
train_prob;

% Done Training

% Testing
% Part 1
correct = 0;
for i = 1:size(test,1)
	test_scores = zeros(num_classes,1);
	for j = 1:num_classes
		test_scores(j) = prior_class(j);
		for k = 1:num_feat
			test_scores(j) = test_scores(j) * train_prob{k}(test(i,k), j);
		end
	end 

	[val, idx] = max(test_scores);
	if(idx == test(i,end))
		correct = correct + 1;
	end
end 

accuracy = correct / size(test, 1)

% Part 3
% Joint estimate of training and testing data

train_log = 0;
for i = 1: size(train, 1) 
	joint_log_prob = log(prior_class(train(i, end)));
	for j = 1:length(train(i, :)) - 1
		joint_log_prob = joint_log_prob + log(train_prob{j}(train(i,j), train(i,end)));		
	end 
	train_log = train_log + joint_log_prob;
end 

train_log

test_log = 0;
for i = 1: size(test, 1) 
	joint_log_prob = log(prior_class(test(i, end)));
	for j = 1:length(test(i, :)) - 1
		joint_log_prob = joint_log_prob + log(train_prob{j}(test(i,j), test(i,end)));		
	end 
	test_log = test_log + joint_log_prob;
end 

test_log

% Part 4
% MAP using dirichlet prior
% for alpha = 1s, MAP = MLE 

train_map_prob = {};

for i = 1:num_feat 
    alpha = 1 * ones(feat_dims(i), 1);
%     alpha(end) = 10;
    map_prob = zeros(feat_dims(i), num_classes);
    for j = 1:feat_dims(i)
        for k = 1:num_classes
            feat_count = length(find(train(:, i) == j & train(:,end) == k));
            class_count = length(find(train(:, end) == k));
            map_prob(j,k) = (feat_count + alpha(j)) / (class_count + sum(alpha));
        end
    end
    train_map_prob = [train_map_prob; {map_prob}];
end

% Testing MAP Estimator
correct = 0;
for i = 1:size(test,1)
	test_scores = zeros(num_classes,1);
	for j = 1:num_classes
		test_scores(j) = prior_class(j);
		for k = 1:num_feat
			test_scores(j) = test_scores(j) * train_map_prob{k}(test(i,k), j);
		end
	end 

	[val, idx] = max(test_scores);
	if(idx == test(i,end))
		correct = correct + 1;
	end
end

accuracy_map = correct / size(test, 1)


% ROC Curve
roc_probs = zeros(size(test,1), 1);
class = 4;
for i = 1:size(test, 1)
    roc_probs(i) = prior_class(class);
%     z = 0;
%     for k = 1:num_classes
%         z = z + prior_class(k)*train_map_prob{j}(test(i, j), k);
%     end
    for j = 1:num_feat
        roc_probs(i) = roc_probs(i) * train_map_prob{j}(test(i,j), class);
    end
    z = 0;
    for k = 1:num_classes
        z_curr = prior_class(k);
        for j = 1:num_feat
            z_curr = z_curr * train_map_prob{j}(test(i,j), k);
        end
        z = z + z_curr;
    end
    roc_probs(i) = roc_probs(i) / z;
end

thresh = 0:0.01:1;
true_poss = zeros(length(thresh), 1);
false_poss = zeros(length(thresh), 1);

% n_class = length(find(test(:, end) == class));

for t = 1:length(thresh)
    class_in = find(roc_probs >= thresh(t));
    class_out = find(roc_probs < thresh(t));
  
    TP = length(find(test(class_in, end) == class));
    FP = length(find(test(class_in, end) ~= class));
    FN = length(find(test(class_out, end) == class));
    TN = length(find(test(class_out, end) ~= class));
    TPR = TP/(TP+FN);
    FPR = FP/(FP+TN);
    
    true_poss(t) = TPR;
    false_poss(t) = FPR;
    
end

plot(false_poss, true_poss);





