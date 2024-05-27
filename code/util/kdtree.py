import numpy as np
from concurrent.futures import ThreadPoolExecutor


class Node:
    """Node class representing a node in the KD tree."""

    def __init__(self, point, left=None, right=None):
        """
        Initialize a Node instance.

        Args:
            point (tuple): The point represented by the node.
            left (Node, optional): The left child node. Defaults to None.
            right (Node, optional): The right child node. Defaults to None.
        """
        self.point = point
        self.left = left
        self.right = right


def build_kdtree(points, depth=0):
    """
    Build a KD tree from a list of points.

    Args:
        points (list): List of points (tuples).
        depth (int, optional): Depth of the current node in the tree. Defaults to 0.

    Returns:
        Node: The root node of the KD tree.
    """
    if len(points) == 0:
        return None

    k = len(points[0])
    axis = depth % k

    points.sort(key=lambda x: x[axis])
    median = len(points) // 2

    return Node(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1 :], depth + 1),
    )


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): First point.
        point2 (tuple): Second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def find_nearest_neighbor(tree, target, depth=0, best=None):
    """
    Find the nearest neighbor to a target point in a KD tree.

    Args:
        tree (Node): The root node of the KD tree.
        target (tuple): The target point.
        depth (int, optional): Depth of the current node in the tree. Defaults to 0.
        best (tuple, optional): The current best neighbor found. Defaults to None.

    Returns:
        tuple: The nearest neighbor to the target point.
    """
    if tree is None:
        return best

    k = len(target)
    axis = depth % k

    if best is None or euclidean_distance(target, tree.point) < euclidean_distance(
        target, best
    ):
        best = tree.point

    if target[axis] < tree.point[axis]:
        next_branch = tree.left
        opposite_branch = tree.right
    else:
        next_branch = tree.right
        opposite_branch = tree.left

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        futures.append(
            executor.submit(find_nearest_neighbor, next_branch, target, depth + 1, best)
        )
        if abs(target[axis] - tree.point[axis]) < euclidean_distance(target, best):
            futures.append(
                executor.submit(
                    find_nearest_neighbor, opposite_branch, target, depth + 1, best
                )
            )

        for future in futures:
            result = future.result()
            if result is not None and euclidean_distance(
                target, result
            ) < euclidean_distance(target, best):
                best = result

    return best
