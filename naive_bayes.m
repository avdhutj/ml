# Naive Bayes classifier assignment

data = load ("nursery_data.txt");

num_feat = 8
num_classes = [3 5 4 4 3 2 3 3 5]

prior_class = zeros(num_classes(num_feat+1), 1);

for i = 1:size(data, 1) 
	prior_class(data(i, num_feat+1)) = prior_class(data(i, num_feat+1)) + 1;
end

prior_class = prior_class / size(data, 1);

prior_class

