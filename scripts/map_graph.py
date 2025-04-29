import json
import math
import sys
import numpy as np
import time
import os
from multiword_name_extraction import extract_map_data_from_all_annotations
sys.path.append(os.pardir + "/scripts")
import coordinate_geometry
from sklearn.linear_model import LogisticRegression
import copy

# can change this constant from one to multiply a padding to the padded_bounding_box attribute
# so far this attribute has not been used.
PADDING_RATIO = 1
        
class FeatureNode:
    def __init__(self, feature_obj):
        self.feature_json = feature_obj
        self.vertices = feature_obj["vertices"]
        self.text = feature_obj["text"]
        self.minimum_bounding_box = coordinate_geometry.convex_hull_min_area_rect(self.vertices)
        self.padded_bounding_box = (self.minimum_bounding_box[0], (PADDING_RATIO * self.minimum_bounding_box[1][0], 
                                            PADDING_RATIO * self.minimum_bounding_box[1][1]), self.minimum_bounding_box[2])
        #print(self.minimum_bounding_box)
        self.neighbors = set()
        self.illegible = feature_obj["illegible"]
        self.truncated = feature_obj["truncated"]
        self.num_letters = len([ch for ch in self.text if ch.isalpha()])
        self.capitalization = self.text.isupper()
    def equals(self, other):
        """Consider two FeatureNodes (i.e. text labels) identical if they have the same text and vertices
        does not override the built-in __eq__ method in order to allow objects to be hashed normally"""
        return (self.text == other.text) and (self.vertices ==  other.vertices)
    def get_ground_truth_linkages(map_filename, annotations_filepath = "rumsey_train.json"):
        """Purpose: create a list of FeatureNodes containing the linkages contained in the ground truth file
        Parameters: map_filename, a string filename of the map image to retrieve the ground truth phrases for
            annotations_filepath, a string of the json file containing the ground truth labels and linkages.
        Returns: a doubly-nested list of phrases for the given map with all groups of labels from the annotations file."""
        phrases_list = []
        map_data = extract_map_data_from_all_annotations(map_filename, annotations_filepath)
        for group in map_data["groups"]:
            cur_list = []
            for label in group:
                cur_node = FeatureNode(label)
                cur_list.append(cur_node)
            phrases_list.append(cur_list)

        return phrases_list
    def get_height(self):
        """
        Gives height dimension of the label's bounding box. If the label is multiple characters long, it is assumed that
        whichever dimension of the bounding box is smaller should be the height dimension.
        Returns: a numerical value representing the height of the feature's minimum bounding box
        """
        if self.num_letters > 1:
            return min(self.minimum_bounding_box[1][0], self.minimum_bounding_box[1][1])
        else:
            return self.minimum_bounding_box[1][1]
    def distance(self, other):
        """
        gives the geographic distance (in meters) between the two polygons, based on the haversine formula distance
        computed for pairs of coordinates on the medians of each side of their minimum bounding boxes
        """
        # get the medians of the four edges of each bounding box
        medians = [[], []]
        rectangles = [self.minimum_bounding_box, other.minimum_bounding_box]
        for i in range(len(rectangles)): 
            # Extract the rectangle properties
            center, size, angle = rectangles[i]
            cx, cy = center
            width, height = size

            # Calculate the medians with rotation consideration
            rotation_rad = np.radians(angle)  # Convert rotation angle to radians

            # Calculate the rotated medians
            medians[i].append([
                cx - 0.5 * width * np.cos(rotation_rad),
                cy - 0.5 * width * np.sin(rotation_rad)
            ])
            medians[i].append([
                cx + 0.5 * width * np.cos(rotation_rad),
                cy + 0.5 * width * np.sin(rotation_rad)
            ])
            medians[i].append([
                cx - 0.5 * height * np.sin(rotation_rad),
                cy + 0.5 * height * np.cos(rotation_rad)
            ])
            medians[i].append([
                cx + 0.5 * height * np.sin(rotation_rad),
                cy - 0.5 * height * np.cos(rotation_rad)
            ])
        distances = [coordinate_geometry.euclidean_distance(s, o) for s in medians[0] for o in medians[1]]
        return min(distances)
    class EdgeCostFunction:
        """Purpose: creates an edge cost function with customized weighting of the features.
        The function takes the form:
        distance*(height_ratio**a)*(1 + b*sin_angle_difference)*(1 + c*capitalization_difference)
        where a, b, and c, are the weights this function is instantiated with"""
        def __init__(self, weights):
            self.a = weights[0]
            self.b = weights[1]
            self.c = weights[2]
        def __call__(self, label1, label2):
            edge_cost =  FeatureNode.distance(label1, label2) * (FeatureNode.height_ratio(label1, label2) ** self.a) 
            edge_cost *= (1 + self.b * FeatureNode.sin_angle_difference(label1, label2)) * (1 + self.c * FeatureNode.capitalization_difference(label1, label2))
            return edge_cost
    def height_difference(self, other):
        return math.fabs(self.get_height() - other.get_height())
    def height_ratio(self, other):
        """
        Purpose: make the height difference feature scale better with the size of the bounding boxes, by returning the ratio of their sizes instead 
        of their absolute value difference."""
        return max(self.get_height() / other.get_height(), other.get_height() / self.get_height())
    def capitalization_difference(self, other):
        """Purpose: represent whether the two labels have different capitalization
        Parameters: self and other, two FeatureNode objects to compare.
        Returns: 0 if the labels are either both in all caps or neither in all caps, 1 otherwise"""
        if self.capitalization != other.capitalization and self.num_letters > 1 and other.num_letters > 1:
            return 1
        return 0
    def distance_height_ratio(self, other):
        return self.distance(other) * self.height_ratio(other)
    def overlays(self, other):
        s_center, s_dims, s_rot = self.minimum_bounding_box
        o_center, o_dims, o_rot = other.minimum_bounding_box
        sx, sy = s_center
        s_width, s_height = s_dims
        ox, oy = o_center
        intersecting = sx + s_width * np.cos(s_rot) / 2 >= ox and sx - s_width * np.cos(s_rot) / 2 <= ox
        intersecting &= sy + s_height * np.sin(s_rot) / 2 >= oy and sy - s_height * np.cos(s_rot) / 2 <= oy
        return intersecting
    def get_angle(self):
        angle = self.minimum_bounding_box[2]
        # rotate the angle by 90 degrees if the "height" dimension of the box is not they y axis.
        if self.get_height() != self.minimum_bounding_box[1][1]:
            angle += 90
        return angle
    def sin_angle_difference(self, other):
        # disregard angle differences for very short words, because they often have high error in 
        # their computed angles
        if self.num_letters <= 2 or other.num_letters <= 2:
            return 0
        return math.sin(math.radians(math.fabs(self.get_angle() - other.get_angle())))
    def distance_height_ratio_sin_angle(self, other):
        return self.distance_height_ratio(other) * (1 + self.sin_angle_difference(other))
    def distance_with_sin_angle_penalty(self, other):
        """
        Returns the distance between the centroids of the two labels, multiplied by 1 plus the sine of their angle difference.
        This is done to discourage matching nearby labels that have very different angled text."""
        return self.distance(other) * (1 + self.sin_angle_difference(other))
    def distance_sin_angle_capitalization_penalty(self, other):
        """
        Returns the distance between the centroids of the two labels, multiplied by 1 plus the sine of their angle difference, 
        and multiplied by2 if the two labels differ in capitalization 
        (i.e. one is all caps and the other contains at least one lower case letter)"""
        coeff = 1
        if self.capitalization != other.capitalization and self.num_letters > 1 and other.num_letters > 1:
            coeff = 2
        return coeff * self.distance(other) * (1 + self.sin_angle_difference(other))
    def distance_height_ratio_sin_angle_capitalization_penalty(self, other):
        return self.distance_sin_angle_capitalization_penalty(other) * self.height_ratio(other)
    
    
    def to_vector(self, weights = [1000, 100, 100]):
        """Purpose: gives a vector representation of the text label
        Returns: a numpy array containing the label center location, height, capitalization, and angle"""
        c = 0
        if self.capitalization:
            c = 1
        return np.array([self.minimum_bounding_box[0][0], self.minimum_bounding_box[0][1], weights[0] * self.get_height(), weights[1] * self.capitalization, c, weights[2] * self.get_angle()])

class EdgeCostFunction:
        """Purpose: creates an edge cost function with customized weighting of the features.
        The function takes the form:
        distance*(height_ratio**a)*(1 + b*sin_angle_difference)*(1 + c*capitalization_difference)
        where a, b, and c, are the weights this function is instantiated with"""
        def __init__(self, weights):
            self.a = weights[0]
            self.b = weights[1]
            self.c = weights[2]
        def __call__(self, label1, label2):
            edge_cost =  FeatureNode.distance(label1, label2) * (FeatureNode.height_ratio(label1, label2) ** self.a) 
            edge_cost *= (1 + self.b * FeatureNode.distance(label1, label2)) * (1 + self.c * FeatureNode.capitalization_difference(label1, label2))
            return edge_cost
class LogisticRegressionEdgeCost:
    def __init__(self, lr_model):
        self.lr_model = lr_model
    def __call__(self, label1, label2):
        attributes = np.array([FeatureNode.distance(label1, label2), FeatureNode.height_ratio(label1, label2) - 1, FeatureNode.sin_angle_difference(label1, label2), FeatureNode.capitalization_difference(label1, label2)])
        return self.lr_model.predict_proba([attributes])[0][0]
class MahalanobisMetric:
        """Compute the edge cost between two features based on the Mahalanobis metric with a learned positive semi-definite matrix
        representing the weights of the attributes"""
        def __init__(self, mmc):
            self.mmc = mmc
        def __call__(self, label1, label2):
            feature_difference_vector = np.array([FeatureNode.distance(label1, label2), FeatureNode.height_ratio(label1, label2) - 1, FeatureNode.sin_angle_difference(label1, label2), FeatureNode.capitalization_difference(label1, label2)])
            distance = self.mmc.pair_distance([[np.array([0,0,0,0]), feature_difference_vector]])[0]
            return distance
def prims_mst(nodes_list, distance_func = FeatureNode.EdgeCostFunction([1, 1, 1])):
    """
    Create a minimum spanning tree of a graph of nodes based on the distance function that is passed
    Parameters: nodes list: a list of Feature Nodes, 
        distance_func: a function to compute the distance between two feature nodes
    Returns: None, modifies the nodes list to connect them into a MST"""
    vertices_list = [{"vertex":node, "key":float("inf"), "parent": None} for node in nodes_list]
    all_neighbors = copy.copy(vertices_list)
    vertices_list[0]["key"] = 0
    while vertices_list != []:
        u = min(vertices_list, key = lambda vertex: vertex["key"])
        vertices_list.remove(u)
        for other_vertex in all_neighbors:
            if other_vertex != u:
                d = distance_func(other_vertex["vertex"], u["vertex"]) 
                if other_vertex in vertices_list and d < other_vertex["key"]:
                    other_vertex["parent"] = u
                    other_vertex["key"] = distance_func(other_vertex["vertex"], u["vertex"])
    for node in all_neighbors:
        if node["parent"] != None:
            node["vertex"].neighbors.add(node["parent"]["vertex"])
            node["parent"]["vertex"].neighbors.add(node["vertex"])
def edge_cut(nodes_list, edge_cost_func = FeatureNode.EdgeCostFunction([1, 1, 1])):
    """
    Purpose: Cut edges from a graph to ensure that each label has degree of at most 2.
    This ensures that every word is considered to be part of at most 1 multi-word phrase.
    Parameters: nodes_list - a list of FeatureNodes linked into a map graph, edge_cost_func -
    a function that takes in two FeatureNodes and measures the (non-negative) edge cost of including them in the graph.
    edge_cut will remove edges that have the highest edge cost until all nodes have at most two edges."""
    nodes_with_more_than_two_edges = {}
    for node in nodes_list:
        if len(node.neighbors) > 2:
            node_edges_list = [(edge_cost_func(node, other_node), other_node) for other_node in node.neighbors]
            nodes_with_more_than_two_edges[node] = node_edges_list
    while len(nodes_with_more_than_two_edges.keys()) > 0:
        # find and delete the highest weight edge connected to a node with more than two neighbors.
        max_weight = 0
        edge_to_cut = None
        for node, edge_list in nodes_with_more_than_two_edges.items():
            for edge in edge_list:
                if edge[0] > max_weight:
                    max_weight = edge[0]
                    edge_to_cut = (node, edge[1])
        # cut the edge
        edge_to_cut[0].neighbors.remove(edge_to_cut[1])
        edge_to_cut[1].neighbors.remove(edge_to_cut[0])
        # update the dictionary of nodes with more than two edges
        if len(edge_to_cut[0].neighbors) <= 2:
            nodes_with_more_than_two_edges.pop(edge_to_cut[0], None)
        else:
            nodes_with_more_than_two_edges[edge_to_cut[0]] = [(edge_cost_func(edge_to_cut[0], other_node), other_node) for other_node in edge_to_cut[0].neighbors]
        if len(edge_to_cut[1].neighbors) <= 2:
            nodes_with_more_than_two_edges.pop(edge_to_cut[1], None)
        else:
            nodes_with_more_than_two_edges[edge_to_cut[1]] = [(edge_cost_func(edge_to_cut[1], other_node), other_node) for other_node in edge_to_cut[1].neighbors]

def cut_cycles(nodes_list, edge_cost_func = FeatureNode.EdgeCostFunction([1, 1, 1])):
    """
    Purpose: Identify cycles in the graph and cut edges in the graph to remove the highest-weighted edge
    in each cycle.
    Parameters: nodes list - a list of FeatureNodes to remove cycles from, edge_cost_func - a function that
    takes two FeatureNode objects and outputs a numerical weight for an edge connecting them."""
    def dfs(node, parent, visited, cycle_edges):
        visited[node] = True
        for neighbor in node.neighbors:
            if not visited[neighbor]:
                if dfs(neighbor, node, visited, cycle_edges):
                    return True
            elif neighbor != parent:  # Found a cycle
                cycle_edges.append((node, neighbor))
                return True
        return False
    def cut_edge_causing_cycle(nodes_list, edge_cost_func):
        visited = {node: False for node in nodes_list}
        cycle_edges = []

        for node in nodes_list:
            if not visited[node]:
                if dfs(node, -1, visited, cycle_edges):
                    break
        # cut the highest-weighted edge in the cycle
        max_weight = -float("inf")
        max_weight_edge = None
        for edge in cycle_edges:
            cur_weight = edge_cost_func(edge[0], edge[1])
            if cur_weight > max_weight:
                max_weight = cur_weight
                max_weight_edge = edge
        if max_weight_edge != None:
            max_weight_edge[0].neighbors.remove(max_weight_edge[1])
            max_weight_edge[1].neighbors.remove(max_weight_edge[0])
            return True
        return False
    counter = 0
    while cut_edge_causing_cycle(nodes_list, edge_cost_func):
        counter += 1
    



def distance_threshold_graph(nodes_list, distance_func = FeatureNode.EdgeCostFunction([0, 0, 0])):
    visited_nodes = set()
    for node in nodes_list:
        visited_nodes.add(node)
        word_length = max(node.minimum_bounding_box[1])
        # gave denominator max of 1 to prevent division by zero errors.
        distance_threshold = 2 * word_length / max(1, len(node.text))
        for other_node in nodes_list:
            if other_node not in visited_nodes and distance_func(node, other_node) <= distance_threshold:
                node.neighbors.add(other_node)
                other_node.neighbors.add(node)
    
                

    
def half_prims_mst(nodes_list, distance_func = FeatureNode.height_difference):
    """
    Create a minimum spanning tree of a graph of nodes based on the distance function that is passed,
    and then removes all of the edges that have a distance that is greater than average
    Parameters: nodes list: a list of Feature Nodes, 
        distance_func: a function to compute the distance between two feature nodes
    Returns: None, modifies the nodes list to connect them into a MST with the half of the edges with the longest
    distances removed"""
    vertices_list = [{"vertex":node, "key":float("inf"), "parent": None} for node in nodes_list]
    all_neighbors = copy.copy(vertices_list)
    vertices_list[0]["key"] = 0
    while vertices_list != []:
        u = min(vertices_list, key = lambda vertex: vertex["key"])
        vertices_list.remove(u)
        for other_vertex in all_neighbors:
            if other_vertex != u:
                d = distance_func(other_vertex["vertex"], u["vertex"]) 
                if other_vertex in vertices_list and d < other_vertex["key"]:
                    other_vertex["parent"] = u
                    other_vertex["key"] = distance_func(other_vertex["vertex"], u["vertex"])
    median_distance = np.median([v["key"] for v in all_neighbors])
    for node in all_neighbors:
        if node["parent"] != None and node["key"] <= median_distance:
            node["vertex"].neighbors.add(node["parent"]["vertex"])
            node["parent"]["vertex"].neighbors.add(node["vertex"])
def spanning_tree_k_neighbors(nodes_list, k = 10):
    vertices_list = [{"index": i,"vertex":node, "key":float("inf"), "parent": None} for i, node in enumerate(nodes_list)]
    all_neighbors = copy.copy(vertices_list)
    vertices_list[0]["key"] = 0
    while vertices_list != []:
        u = min(vertices_list, key = lambda vertex: vertex["key"])
        vertices_list.remove(u)
        index_range = (max(u["index"] - k // 2, 0), min(u["index"] + k // 2, len(all_neighbors)))
        for other_vertex in all_neighbors[index_range[0]:index_range[1]]:
            if other_vertex != u:
                if other_vertex in all_neighbors and other_vertex["vertex"].distance(u["vertex"]) < other_vertex["key"]:
                    other_vertex["parent"] = u
                    other_vertex["key"] = other_vertex["vertex"].distance(u["vertex"])
    for node in all_neighbors:
        if node["parent"] != None:
            node["vertex"].neighbors.add(node["parent"]["vertex"])
            node["parent"]["vertex"].neighbors.add(node["vertex"])
def draw_complete_graph(nodes_list):
    """
    Purpose: draw edges between each vertex of the graph and all other vertices of the graph."""
    for node in nodes_list:
        for other_node in nodes_list:
            if not (node is other_node):
                node.neighbors.add(other_node)
                other_node.neighbors.add(other_node)
def connect_with_rf_classifier(nodes_list, rf_classifer):
    visited_nodes = set()
    for node in nodes_list:
        visited_nodes.add(node)
        for other_node in nodes_list:
            if other_node not in visited_nodes:
                features = [[node.distance(other_node), node.height_ratio(other_node), node.sin_angle_difference(other_node), node.capitalization_difference(other_node)]]
                predicted_connectedness = rf_classifer.predict(features)
                if predicted_connectedness[0] == 1:
                    # draw edges between the two nodes if the rf_classifier predicts 1
                    node.neighbors.add(other_node)
                    other_node.neighbors.add(node) 
class MapGraph:
    def __init__(self, map_filename = None, connecting_function = None, annotations_filepath = "rumsey_train.json"):
        # to draw the edges for the graph included in the annotated data, set connecting_function parameter to "annotations"
        self.nodes = []
        self.map_filename = map_filename
        if map_filename != None:
            map_data = extract_map_data_from_all_annotations(map_filename, annotations_filepath)
            for group in map_data["groups"]:
                prev_node = None
                for label in group:
                    cur_node = FeatureNode(label)
                    self.nodes.append(cur_node)
                    if connecting_function == "annotations" and prev_node != None:
                        prev_node.neighbors.add(cur_node)
                        cur_node.neighbors.add(prev_node)
                    prev_node = cur_node

            #print("Time loading:", time.time() - time_loading)
            if connecting_function != None and not isinstance(connecting_function, str):
                #time_connecting = time.time()
                connecting_function(self.nodes)
                #print("Time connecting:", time.time() - time_connecting)
    def __repr__(self):
        sting_representation = ""
        for node in self.nodes:
            sting_representation += node.text + "\n"
            for neighbor in node.neighbors:
                sting_representation += "    " + neighbor.text + "\n"
        return sting_representation
    def count_edges(self):
        count = 0
        for node in self.nodes:
            count += len(node.neighbors)
        return count / 2
    def to_matrix(nodes_list, weights = [1000, 100, 100]):
        """
        Purpose: represent all of the nodes in the graph as a matrix
        Returns: a matrix containing the vector representation of each node"""
        return np.array([node.to_vector(weights) for node in nodes_list])
    def __contains__(self, node):
        for graph_node in self.nodes:
            if node.equals(graph_node):
                return True
        return False
    def sort_by_geometry(self):
        self.nodes.sort(key=lambda x : x.minimum_bounding_box[0][1])
        self.nodes.sort(key=lambda x : x.minimum_bounding_box[0][0])
    def to_json(self):
        """
        Purpose: Converts the map graph into a json object in the format used in the ICDAR Map Text Competition.
        IMPORTANT: this method only works for graphs where none of the nodes have a degree higher than 3.
        Otherwise, some word(s) will be considered to be part of multiple different phrases, which does not
        match with the competition format."""
        # sort by y coordinate and then 
        self.sort_by_geometry()
        json_representation = {"image":self.map_filename, "groups":[]}
        remaining_nodes = self.nodes[:]
        words_in_json = 0
        while len(remaining_nodes) > 0:
            # find the first word that is not an interior word of a phrase
            index = 0
            while index < len(remaining_nodes) and len(remaining_nodes[index].neighbors) >= 2:
                index += 1
            current_group = [remaining_nodes[index].feature_json]
            # remove the element added to the group
            starting_node = remaining_nodes.pop(index)
            # if the element had no neighbors, move on to the next iteration
            # otherwise, keep adding connected elements to its group
            if len(starting_node.neighbors) == 1:
                # add the first neighbor to the group and find the neighbor's other neighbors
                current_neighbor = list(starting_node.neighbors)[0]
                current_group.append(current_neighbor.feature_json)
                neighbors_list = copy.copy(current_neighbor.neighbors)
                remaining_nodes.remove(current_neighbor)
                neighbors_list.remove(starting_node)
                while len(neighbors_list) >= 1:
                    prev_neighbor = current_neighbor
                    current_neighbor = list(neighbors_list)[0]
                    current_group.append(current_neighbor.feature_json)
                    neighbors_list = copy.copy(current_neighbor.neighbors)
                    remaining_nodes.remove(current_neighbor)
                    neighbors_list.remove(prev_neighbor)
            json_representation["groups"].append(current_group)
            words_in_json += len(current_group)
        #print(words_in_json)
        return json_representation
if __name__ == "__main__":
    mg = MapGraph("rumsey/test/3081001_h6_w18.png", prims_mst, "test_annotations.json")
    edge_cut(mg.nodes)
    mg.to_json()





            

        

