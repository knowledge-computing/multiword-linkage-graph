import os
import sys
sys.path.append(os.pardir)
import json
import random
class AnnotationsSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self):
        if not hasattr(self, 'annotations_dict'):
            print("initializing dict")
            self.annotations_dict = {}
    def get_annotations(self, annotations_filepath):
        if annotations_filepath in self.annotations_dict:
            return self.annotations_dict[annotations_filepath]
        else:
            print("loading annotations")
            with open(annotations_filepath, "r") as f:
                self.annotations_dict[annotations_filepath] = json.load(f)
                return self.annotations_dict[annotations_filepath]


def extract_map_data_from_all_annotations(map_filename, annotations_filepath = "rumsey_train.json"):
    map_data = None
    for map in AnnotationsSingleton().get_annotations(annotations_filepath):
        if map["image"] == map_filename:
            return map
    if (map_data == None):
        print(map_filename + " not found")
        return None
def find_multiword_phrases_in_map(map_data):
    names = []
    for group in map_data["groups"]:
        # print(group)
        cur_name = ""
        for label in group:
            cur_name += label["text"] + " "
        if (cur_name.count(" ") >= 2):
            names.append(cur_name[:-1])
    return names
def list_phrases_from_map(map_data):
    """
    This function is different from find_multiword_phrases_in_map in the fact that it includes single word phrases as well
    as multiword phrases. The output format is also a nested list, where each inner list is a list of individual words in 
    a phrase, rather than a list of complete phrases."""
    phrases = []
    for group in map_data["groups"]:
        # print(group)
        cur_phrase = []
        for label in group:
            cur_phrase.append(label["text"])
        phrases.append(cur_phrase)
    return phrases
def remove_groupings_from_annotations(map_annotations_dict):
    removed_dict_annotations = {"image":map_annotations_dict["image"], "groups":[]}
    for group in map_annotations_dict["groups"]:
        for label in group:
            removed_dict_annotations["groups"].append(label)
    return removed_dict_annotations
def list_all_word_labels(annotations_dict):
    word_labels = []
    for group in annotations_dict["groups"]:
        for label in group:
            word_labels.append(label["text"])
    return word_labels

def multiword_name_extraction_from_map(map_filename, data_filepath = "rumsey_train.json"):
    map_data = extract_map_data_from_all_annotations(map_filename, data_filepath)
    return find_multiword_phrases_in_map(map_data)

