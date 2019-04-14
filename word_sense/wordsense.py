import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import numpy as np
from numpy import dot
from numpy import average
from numpy.linalg import norm


class WordSense(object):
    """docstring for WordSense"""

    def __init__(self, glove_file_path, sentence_to_disambiguate, threshold):
        self.__cosine_sim_threshold = threshold
        self.__pos_vectors = {}
        self.__successful_replaced_words = {}
        self.__sense_vectors_collection = {}
        self.__sorted_sense_vectors_collection = {}
        self.__glove = self.__load_glove_vectors(glove_file_path)
        self.__initialize_sorted_sense_vector(sentence_to_disambiguate)

    def __initialize_sorted_sense_vector(self, sentence_to_disambiguate):
        tokens_input = nltk.word_tokenize(sentence_to_disambiguate)
        pos_tags_input = nltk.pos_tag(tokens_input)

        pos = []
        for word, pos_tag in pos_tags_input:
            # print(word, "is tagged as", pos_tag)
            # if pos_tag == 'VBD' or pos_tag == 'NN' or pos_tag == 'ADJ' or pos_tag == 'ADV':
            try:
                self.__pos_vectors[word] = self.__glove[word]
                pos.append(word)
            except Exception:
                # print(pos, " not found in glove")
                pass
        for candidate in pos:
            sense_vectors = self.__get_word_sense_vectors(candidate)
            if sense_vectors is None:
                continue
            self.__sorted_sense_vectors_collection[candidate] = len(sense_vectors)
            self.__sense_vectors_collection[candidate] = sense_vectors

        self.__sorted_sense_vectors_collection = sorted(self.__sorted_sense_vectors_collection.items(),
                                                        key=lambda x: x[1])

    @staticmethod
    def __load_glove_vectors(glove_file):
        with open(glove_file, 'r', encoding='utf-8') as file:
            vectors = {}
            for line in file:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                vectors[word] = embedding
            return vectors

    def __get_word_sense_vectors(self, candidate):
        vectors = {}
        try:
            candidate_vec = self.__glove[candidate]
        except Exception:
            # print(candidate, " not found in glove")
            return None
        for ss in wn.synsets(candidate):
            # print("synonym of ", candidate, " is ", ss.lemmas()[0].name())
            tokens = nltk.word_tokenize(ss.definition())
            pos_tags = nltk.pos_tag(tokens)
            word_vectors = []
            for gloss_pos, tag in pos_tags:
                # if tag == 'VBD' or tag == 'NN' or tag == 'ADJ' or tag == 'ADV':
                try:
                    gloss_word_vec = self.__glove[gloss_pos]
                except Exception:
                    # print(gloss_pos, " not found in glove")
                    continue
                cos_sim = dot(gloss_word_vec, candidate_vec) / (norm(gloss_word_vec) * norm(candidate_vec))
                if cos_sim > self.__cosine_sim_threshold:
                    word_vectors.append(gloss_word_vec)
            if len(word_vectors) == 0:
                continue
            sense_vector = average(word_vectors)
            vectors[ss] = sense_vector
        return vectors

    def __disambiguate_word_sense(self, word, context_vector):
        vectors = self.__sense_vectors_collection[word]
        cos_sims = {}
        for sense, sense_vector in vectors.items():
            cos_sim = dot(context_vector, sense_vector) / (norm(context_vector) * norm(sense_vector))
            cos_sims[sense.lemmas()[0].name()] = cos_sim
        sorted_list = sorted(cos_sims.items(), key=lambda x: x[1])
        if len(sorted_list) == 0:
            return None
        nearest_sense_word = sorted_list.pop()[0]
        return nearest_sense_word

    def get_successful_replaced_words(self):
        context_vec = average(list(self.__pos_vectors.values()))
        for w, _ in self.__sorted_sense_vectors_collection:
            nearest_word = self.__disambiguate_word_sense(w, context_vec)
            try:
                self.__pos_vectors[nearest_word] = self.__glove[nearest_word]
                self.__successful_replaced_words[w] = nearest_word
                print(w, " is replaced with ", nearest_word)
                if nearest_word != w:
                    self.__pos_vectors.pop(w)
                context_vec = average(list(self.__pos_vectors.values()))
            except Exception as e:
                # print(nearest_word, " not found in glove")
                continue
        return self.__successful_replaced_words


wordSense = WordSense(glove_file_path='C:/Users/logicalfarhad/Downloads/glove.6B/glove.6B.50d.txt',
                      sentence_to_disambiguate='quick brown fox jumps over the lazy dog', threshold=0.05)
print(wordSense.get_successful_replaced_words())
