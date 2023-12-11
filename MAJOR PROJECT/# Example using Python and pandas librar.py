# Example using Python and pandas library for collaborative filtering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD  # Surprise library for collaborative filtering

# Load movie ratings data
ratings_data = pd.read_csv('ratings.csv')  # Assuming 'ratings.csv' contains user-item ratings data

# Prepare data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)

# Split data into train and test sets
train_set, test_set = train_test_split(data, test_size=0.2)

# Train the SVD algorithm (Singular Value Decomposition)
algo = SVD()
algo.fit(train_set)

# Predict ratings for test data
predictions = algo.test(test_set)

# Evaluate model performance
rmse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions], squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')
