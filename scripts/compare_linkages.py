import os
import random
import sys
sys.path.append(os.pardir)
import list_multiword_paths
import multiword_name_extraction
from map_graph import FeatureNode, prims_mst, half_prims_mst, distance_threshold_graph, MapGraph
import json
import os
import numpy as np
from nltk.cluster.util import euclidean_distance
import multiprocessing
from clustering_linkages import cluster_map_nodes, distance_height_ratio_sin_angle_capitalization
from cross_validation_experiment import load_folds_from_folder

class LinkageMethod:
    def connect(self, map_graph):
        if self.weights == None:
            self.connection_function(map_graph.nodes, self.distance_function)
        else:
            self.connection_function(map_graph.nodes, self.weights, self.distance_function)
    def __init__(self, connection_function, distance_function, weights = None):
        self.connection_function = connection_function
        self.distance_function = distance_function
        self.weights = weights 


def compare_linkages(map_filename, annotations_filepath = "rumsey_train.json", linkage_method = LinkageMethod(prims_mst, FeatureNode.distance)):
    ground_truth = FeatureNode.get_ground_truth_linkages(map_filename, annotations_filepath)
    linkages = MapGraph(map_filename, annotations_filepath=annotations_filepath)
    linkage_method.connect(linkages)
    correctly_linked_phrases = []
    incorrectly_linked_phrases = []
    single_word_phrases = []
    number_correct_edges = 0
    for annotated_phrase in ground_truth:
        if len(annotated_phrase) > 1:
            found = list_multiword_paths.search_node_sequence_in_graph(linkages.nodes, annotated_phrase)
            if not found:
                incorrectly_linked_phrases.append(list_multiword_paths.express_node_sequence_as_phrase(annotated_phrase))
            else:
                correctly_linked_phrases.append(list_multiword_paths.express_node_sequence_as_phrase(annotated_phrase))
                number_correct_edges += len(annotated_phrase) - 1
        else:
            single_word_phrases.append(annotated_phrase)
    results = {"image":map_filename, "results":
                    {
                        "correctly_linked_phrases": correctly_linked_phrases,
                        "incorrectly_linked_phrases": incorrectly_linked_phrases,
                        "number_correctly_linked": len(correctly_linked_phrases),
                        "number_connections_missed": len(ground_truth) - len(correctly_linked_phrases) - len(single_word_phrases),
                        "total_number_edges": linkages.count_edges(),
                        "number_correct_edges": number_correct_edges

                    }
               }
    return results
def map_list_compare_linkages(map_list, function_name, annotations_filepath = "rumsey_train.json", linkage_method = LinkageMethod(prims_mst, FeatureNode.distance), output_filepath = None):
    results_dict = {"connecting_method": "MST", "distance_function":function_name, "map_results":[]}
    for map_id in map_list:
        results_dict["map_results"].append(compare_linkages(map_id, annotations_filepath, linkage_method))
    if output_filepath != None:
        with open(output_filepath, "w") as f:
            json.dump(results_dict, f)
    print("finished", function_name)
    return results_dict
    

def get_stats_from_results_file(results_filename, annotations_filepath = "rumsey_train.json"):
    with open(results_filename) as f:
        results_dict = json.load(f)
        true_linkages = []
        missed_linkages = []
        false_linkages = []
        sum_correct_edge_count = 0
        sum_edge_count = 0
        for map_result in results_dict["map_results"]:
            # "number_correctly_linked": 3, "number_connections_missed": 125, "number_incorrectly_linked": 401}
            #true_linkages.append(map_result["results"]["number_correctly_linked"])
            num_correctly_linked = 0
            correct_edge_count = 0
            for phrase in map_result["results"]["correctly_linked_phrases"]:
                # filter out phrases that are single words
                # from the correctly linked results
                # as single word phrases would always be correctly linked as they don't need
                # to be linked
                if phrase.count(" ") > 0:
                    num_correctly_linked += 1
                    if "number_correct_edges" not in map_result["results"]:
                        correct_edge_count += phrase.count(" ")
            if "number_correct_edges" in map_result["results"]:
                correct_edge_count = map_result["results"]["number_correct_edges"]
            if "total_number_edges" in map_result["results"]:
                sum_edge_count += map_result["results"]["total_number_edges"]
            else:
                sum_edge_count += len(multiword_name_extraction.list_all_word_labels(multiword_name_extraction.extract_map_data_from_all_annotations(map_result["image"], annotations_filepath))) - 1  
            sum_correct_edge_count += correct_edge_count
            true_linkages.append(num_correctly_linked)
            #false_linkages.append(map_result["results"]["number_incorrectly_linked"])
            missed_linkages.append(map_result["results"]["number_connections_missed"])
        recall_per_map = [true_linkages[i] / max(1, (true_linkages[i] + missed_linkages[i])) for i in range(len(true_linkages))]
        print(recall_per_map)
        #precision_per_map = [true_linkages[i] / (true_linkages[i] + false_linkages[i]) for i in range(len(true_linkages))]
        print("recall of phrases: ", sum(true_linkages)/sum(true_linkages + missed_linkages))
        print("best recall on map", max(recall_per_map), "\nworst recall on map", min(recall_per_map))
        #print("best precision on map", max(precision_per_map), "\nworst precision on map", min(precision_per_map))
        print("precision of edges: ", sum_correct_edge_count / sum_edge_count)
    return (np.mean(recall_per_map))
