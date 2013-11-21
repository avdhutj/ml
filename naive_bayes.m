# Naive Bayes classifier assignment

data = load ("nursery_data.txt");

num_feat = 8;
feat_dims = [3 5 4 4 3 2 3 3];
num_classes = 5;

#reset(RandStream.getDefaultStream);
data = data(randperm(size(data,1)), :);
train = data(1:size(data, 1)/2, :);
test = data(size(data,1)/2+1:end,:);

# Training
prior_class = zeros(num_classes, 1);

for i = 1:size(train, 1) 
	prior_class(train(i, end)) = prior_class(train(i, end)) + 1;
end

prior_class = prior_class / size(train, 1);

train_prob = {};

for i = 1:num_feat
	feat_train_prob = zeros(num_classes, feat_dims(i));
	for j = 1 : feat_dims(i)
		for k = 1 : num_classes
			feat_count = length(find(train(:,i) == j & train(:,end) == k ));
			class_count = length(find(train(:,i) == k));
			feat_train_prob(j,k) = (feat_count + 1 )/ (class_count + 1);  # With Laplace Smoothning
		end
	end
	train_prob = [train_prob; {feat_train_prob}];
end


prior_class;
train_prob;

# Done Training

# Testing
correct = 0;
for i = 1:size(test,1)
	test_scores = zeros(num_classes,1);
	for j = 1:num_classes
		test_scores(j) = prior_class(j);
		for k = length(test(i,:))-1
			test_scores(j) = test_scores(j) * train_prob{k}(test(i,k), j);
		end
	end 

	[val, idx] = max(test_scores);
	if(idx == test(i,end))
		correct = correct + 1;
	end
end 

accuracy = correct / size(test, 1)
