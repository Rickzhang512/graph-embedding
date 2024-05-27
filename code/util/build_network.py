import networkx as nx
import pandas as pd


def preprocess_dataset(dataset_path):
    """
    Preprocess the original dataset containing user-song interactions.

    Args:
    - dataset_path (str): Path to the dataset file containing user-song interactions.

    Returns:
    - preprocessed_df (pandas.DataFrame): Preprocessed DataFrame with columns: 'user_id', 'song_id', 'interaction_type'.
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Drop any rows with missing values
    df = df.dropna()

    # Ensure 'user_id' and 'song_id' are string type
    df["user_id"] = df["user_id"].astype(str)
    df["song_id"] = df["song_id"].astype(str)

    # Ensure 'interaction_type' is lowercase
    df["interaction_type"] = df["interaction_type"].str.lower()

    return df


def generate_interaction_network(dataset_path):
    """
    Generate a network of user-song interactions from a given dataset.

    Args:
    - dataset_path (str): Path to the dataset file containing user-song interactions.

    Returns:
    - G (networkx.Graph): Network representing user-song interactions.
    """
    # Preprocess dataset
    df = preprocess_dataset(dataset_path)

    # Create an empty graph
    G = nx.Graph()

    # Add nodes for users and songs
    users = set(df["user_id"])
    songs = set(df["song_id"])
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(songs, bipartite=1)

    # Add edges for user-song interactions
    for _, row in df.iterrows():
        user_id, song_id, interaction_type = (
            row["user_id"],
            row["song_id"],
            row["interaction_type"],
        )
        G.add_edge(user_id, song_id, interaction_type=interaction_type)

    return G
