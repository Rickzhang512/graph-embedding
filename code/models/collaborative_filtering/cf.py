import numpy as np


class CollaborativeFiltering:
    def __init__(self, data):
        """
        Initialize CollaborativeFiltering with user-item rating data.

        Parameters:
        - data (dict): A dictionary where keys are user identifiers and values are dictionaries
                       mapping item identifiers to ratings given by the corresponding user.
        """
        self.data = data

    def calculate_similarity(self, user1, user2):
        """
        Calculate similarity between two users using Pearson correlation coefficient.

        Parameters:
        - user1 (str): Identifier of the first user.
        - user2 (str): Identifier of the second user.

        Returns:
        - similarity (float): Pearson correlation coefficient representing the similarity
                              between the two users' ratings.
        """
        ratings1 = np.array(
            [self.data[user1].get(item, 0) for item in self.data[user2]]
        )
        ratings2 = np.array(
            [self.data[user2].get(item, 0) for item in self.data[user1]]
        )

        common_items_mask = (ratings1 != 0) & (ratings2 != 0)
        if not np.any(common_items_mask):
            return 0

        ratings1 = ratings1[common_items_mask]
        ratings2 = ratings2[common_items_mask]

        mean1 = np.mean(ratings1)
        mean2 = np.mean(ratings2)

        num = np.sum((ratings1 - mean1) * (ratings2 - mean2))
        den = np.sqrt(np.sum((ratings1 - mean1) ** 2) * np.sum((ratings2 - mean2) ** 2))

        if den == 0:
            return 0

        return num / den

    def recommend(self, user, num_recommendations=5):
        """
        Recommend items to a user.

        Parameters:
        - user (str): Identifier of the user for whom recommendations are generated.
        - num_recommendations (int): Number of recommendations to generate (default is 5).

        Returns:
        - recommendations (list): A list of tuples containing recommended items and their scores,
                                  sorted in descending order of score.
        """
        scores = {}
        for other_user in self.data:
            if other_user != user:
                similarity = self.calculate_similarity(user, other_user)

                for item in self.data[other_user]:
                    if item not in self.data[user] or self.data[user][item] == 0:
                        scores.setdefault(item, 0)
                        scores[item] += self.data[other_user][item] * similarity

        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :num_recommendations
        ]
        return recommendations

    def evaluate(self, test_data, num_recommendations=5):
        """
        Evaluate the recommendations using MAP and NDCG.

        Parameters:
        - test_data (dict): A dictionary containing test user-item data for evaluation.
                            Format is similar to the input data dictionary.
        - num_recommendations (int): Number of recommendations to generate for evaluation (default is 5).

        Returns:
        - map_score (float): Mean Average Precision (MAP) score.
        - ndcg_score (float): Normalized Discounted Cumulative Gain (NDCG) score.
        """
        average_precision = 0
        ndcg = 0
        num_users = len(test_data)

        for user in test_data:
            recommended_items = [
                item for item, _ in self.recommend(user, num_recommendations)
            ]
            actual_items = [
                item for item in test_data[user] if test_data[user][item] > 0
            ]

            # Calculate Average Precision
            precision = 0
            num_hits = 0
            for i, item in enumerate(recommended_items):
                if item in actual_items:
                    num_hits += 1
                    precision += num_hits / (i + 1)
            average_precision += precision / min(len(actual_items), num_recommendations)

            # Calculate NDCG
            dcg = 0
            idcg = sum(
                1 / (i + 1) for i in range(min(len(actual_items), num_recommendations))
            )
            for i, item in enumerate(recommended_items):
                if item in actual_items:
                    relevance = 1
                    rank = i + 1
                    dcg += (pow(2, relevance) - 1) / (np.log2(rank + 1))
            ndcg += dcg / idcg

        map_score = average_precision / num_users
        ndcg_score = ndcg / num_users

        return map_score, ndcg_score
