# Naive Bayes classifier assignment

data = load ("nursery_data.txt");

num_feat = 8
num_classes = [3 5 4 4 3 2 3 3 5]

#reset(RandStream.getDefaultStream);
data = data(randperm(size(data,1)), :);
train = data(1:size(data, 1)/2, :);
test = data(size(data,1)/2+1:end,:);

# Training
prior_class = zeros(num_classes(num_feat+1), 1);

for i = 1:size(train, 1) 
	prior_class(train(i, num_feat+1)) = prior_class(train(i, num_feat+1)) + 1;
end

prior_class = prior_class / size(train, 1);

prior_class

# Done Training
