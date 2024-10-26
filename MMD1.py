import numpy as np

def center_and_nan_to_zero(matrix, axis=1):
    """ Center the matrix and replace nan values with zeros"""
    means = np.nanmean(matrix, axis=axis, keepdims=True)
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)

def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def fast_cosine_sim(utility_matrix, vector, axis=1):
    """ Compute the cosine similarity between the matrix and the vector"""
    norms = np.linalg.norm(utility_matrix, axis=axis, keepdims=True)
    um_normalized = utility_matrix / norms
    dot = np.dot(um_normalized, vector)
    scaled = dot / np.linalg.norm(vector)
    return scaled

def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    print("Clean Utility Matrix:\n", clean_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_ratings = clean_utility_matrix[user_index, :]
    print(f"User Row (User {user_index} Ratings):\n", user_ratings)
    similarities = fast_cosine_sim(clean_utility_matrix, user_ratings)
    print("Similarities:\n", similarities)
    
    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[user_index, item_index]):
            return orig_utility_matrix[user_index, item_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(~np.isnan(orig_utility_matrix[:, item_index]))[0]
        # Get indices of users with the highest similarity
        best_among_who_rated = np.argsort(similarities[users_who_rated])  # Result indices are relative to users_who_rated
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[~np.isnan(similarities[best_among_who_rated])]
        
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            weighted_sum = np.dot(similarities[best_among_who_rated],
                                  orig_utility_matrix[best_among_who_rated, item_index])
            sum_of_similarities = np.sum(similarities[best_among_who_rated])
            rating_of_item = weighted_sum / sum_of_similarities  # Complete code here
        else:
            rating_of_item = np.nan
        
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[1]  # Change to get number of items

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings



def test():
    orig_utility_matrix = np.array([
        #Item 1 | Item 2 | Item 3 | Item 4
        [1, 2, 3, 4],   # User 0
        [5, np.nan, 6, 7],   # User 1
        [8, np.nan, 9, 10],  # User 2
        [11, np.nan, 13, 14]  # User 3
    ])
    user_index = 3
    neighborhood_size = 3 
    print("Input Matrix:\n", orig_utility_matrix)

    user_index = 3
    neighborhood_size = 3 

    predicted_ratings = rate_all_items(orig_utility_matrix, user_index, neighborhood_size)

    print("Predicted Ratings for User", user_index, ":", predicted_ratings)
    print("Expected Ratings for User", user_index, ": [11.0, 2.0, 13.0, 14.0]")

