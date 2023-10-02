from surprise import Dataset
from surprise import KNNBasic
from surprise import Reader

# Load data
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('ratings.csv', reader=reader)

# Split data into train and test sets
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

# Define and train the model
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Get top recommendations for a given user
user_id = 1
n_recommendations = 10
user_ratings = trainset.ur[user_id]
movies_seen = [item_id for (item_id, rating) in user_ratings]
candidates = [item for item in trainset.all_items() if item not in movies_seen]
predictions = [model.predict(str(user_id), str(item_id)) for item_id in candidates]
top_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations]

# Print top recommendations
for recommendation in top_recommendations:
    print('Movie:', recommendation.iid, 'Estimated rating:', recommendation.est)
