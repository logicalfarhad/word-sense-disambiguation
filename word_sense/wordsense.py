import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import numpy as np
from numpy import dot
from numpy import average
from numpy.linalg import norm
import os


def load_glove_vectors(glove_file):
    f = open(glove_file, 'r', encoding="utf-8")
    vectors = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        vectors[word] = embedding
    f.close()
    return vectors


cosine_sim_threshold = 0.05
score_margin_threshold = 0.1


def get_valid_pos_tag(tag):
    if tag.startswith('J') or tag.startswith('V') or tag.startswith('N') or tag.startswith('R'):
        return True
    return False


def get_word_sense_vectors(candidate):
    vectors = {}
    try:
        candidate_vec = glove[candidate]
    except Exception:
        # print(candidate, "not found in glove")
        return None
    for sense in wn.lemmas(candidate):
        # if candidate == "bank":
        # print("synonym of ", candidate, " is ", ss.lemmas()[0].name())
        # print("key of ", candidate, " is ", ss.lemmas()[0].key())
        gloss = [sense.synset().definition()]
        gloss.extend(sense.synset().examples())
        word_vectors = []
        for sentence in gloss:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            for gloss_pos, tag in pos_tags:
                if get_valid_pos_tag(tag):
                    try:
                        gloss_word_vec = glove[gloss_pos]
                    except Exception:
                        # print(gloss_pos, "not found in glove")
                        continue
                    cos_sim = dot(gloss_word_vec, candidate_vec) / (norm(gloss_word_vec) * norm(candidate_vec))
                    if cos_sim > cosine_sim_threshold:
                        word_vectors.append(gloss_word_vec)
        if len(word_vectors) == 0:
            continue
        sense_vector = average(word_vectors, 0)
        vectors[sense] = sense_vector
    return vectors


def disambiguate_word_sense(word, context_vector):
    vectors = sense_vectors_collection[word]
    if len(vectors) == 0:
        return [None, 0.0]
    cos_sims = {}
    for sense, sense_vector in vectors.items():
        cos_sim = dot(context_vector, sense_vector) / (norm(context_vector) * norm(sense_vector))
        cos_sims[sense] = cos_sim
    sorted_list = sorted(cos_sims.items(), key=lambda x: x[1])
    if len(sorted_list) == 0:
        return [None, 0.0]
    most_similar_pair = sorted_list.pop()
    disambiguated_sense = most_similar_pair[0]
    cos_sim_second_most_similar_sense = 0
    if len(sorted_list) > 0:
        cos_sim_second_most_similar_sense = sorted_list.pop()[1]
    score_margin = most_similar_pair[1] - cos_sim_second_most_similar_sense
    # we return the disambiguated sense AND the cosine score margin between the two most similar senses.
    return [disambiguated_sense, score_margin]


# glove = load_glove_vectors('/media/iftekhar/New Volume/Personal/Admission Docs/Germany/RWTH/MI/Lab - AI_Language_Technology/training_nball47634/glove.6B.50d.txt')
# glove = load_glove_vectors('/media/iftekhar/New Volume/Personal/Admission Docs/Germany/RWTH/MI/Lab - AI_Language_Technology/deps.words')
# glove = load_glove_vectors('/media/iftekhar/New Volume/Personal/Admission Docs/Germany/RWTH/MI/Lab - AI_Language_Technology/bow2.words')
# glove = load_glove_vectors('/media/iftekhar/New Volume/Personal/Admission Docs/Germany/RWTH/MI/Lab - AI_Language_Technology/bow5.words')
glove = load_glove_vectors('E:/Code/deps.words/deps.words')
sense_vectors_collection = {}

annotation_results = dict()


def find_wn_key(sentence, lookup_word):
    sorted_sense_vectors_collection = {}
    pos = []
    pos_vectors = {}
    tokens_input = nltk.word_tokenize(sentence)
    pos_tags_input = nltk.pos_tag(tokens_input)
    for word, pos_tag in pos_tags_input:
        # print(word, "is tagged as", pos_tag)
        if get_valid_pos_tag(pos_tag):
            try:
                pos_vectors[word] = glove[word]
                pos.append(word)
            except Exception:
                pass
                # print(pos, " not found in glove")
    for p in pos:
        sense_vectors = get_word_sense_vectors(p)
        if sense_vectors is None:
            continue
        sense_vectors_collection[p] = sense_vectors
        sorted_sense_vectors_collection[p] = len(sense_vectors)
    # S2C sorting for content word
    sorted_sense_vectors_collection = sorted(sorted_sense_vectors_collection.items(), key=lambda x: x[1])
    # print("sorted by sense count", sorted_sense_vectors_collection)
    # Context vector initialization
    context_vec = average(list(pos_vectors.values()), 0)
    wn_key = "not found"
    for w, _ in sorted_sense_vectors_collection:
        disambiguation_results = disambiguate_word_sense(w, context_vec)
        disambiguated_sense = disambiguation_results[0]
        if disambiguated_sense is None:
            continue
        if w == lookup_word:
            wn_key = disambiguated_sense._key
            break
        score_margin = disambiguation_results[1]
        if score_margin > score_margin_threshold:
            pos_vectors[w] = sense_vectors_collection[w][disambiguated_sense]
            context_vec = average(list(pos_vectors.values()), 0)
    # print(pos_vectors.keys())
    sense_vectors_collection.clear()
    return wn_key


def load_annotations():
    path = "E:/Code/hlt2005releasev2/hlt2005releasev2/domainhltGS.tar/gold_standard_clean.txt"
    with open(path, 'r', encoding='ISO-8859-1') as f:
        for lines in f:
            line = lines.split('|')
            # print(line)
            if len(line) < 4:
                continue
            linkup_key = line[1].strip()
            # print("linkup key", linkup_key)
            wn_key = line[2].strip()
            # print("wn key", wn_key)
            wn_keylist = list()
            if linkup_key in annotation_results:
                wn_keylist = annotation_results[linkup_key]
            else:
                annotation_results[linkup_key] = wn_keylist
            if wn_key == "unclear":
                continue
            wn_keylist.append(wn_key)


# output_file = open("output_bow2_windows.txt", "w")
# results_file = open("wsd_results_bow2_windows.txt", "w")
# output_file = open("output_bow5_windows.txt", "w")
# results_file = open("wsd_results_bow5_windows.txt", "w")
# output_file = open("output_glove.6B.50d_windows.txt", "w")
# results_file = open("wsd_results_glove.6B.50d_windows.txt", "w")
output_file = open("output_deps_windows.txt", "w")
results_file = open("wsd_results_deps_windows.txt", "w")

load_annotations()

correct_count = 0
invalid_linkup_key_count = 0
total_sentence_count = 0

for dirpath, _, filenames in os.walk("E:/Code/hlt2005releasev2/hlt2005releasev2/domainhltGS.tar/Annotated_Sentences"):
    if len(filenames) == 0:
        continue
    for file in filenames:
        f = open(os.path.join(dirpath, file), 'r', encoding='ISO-8859-1')
        #from itertools import islice

        #for line in islice(f, 1):
        for line in f:
            split_line = line.split('?')
            metadata_array = split_line[0].split(' ')
            linkup_key = metadata_array[0]
            lookup_word = metadata_array[2]
            sentence = split_line[1].split(' ', 2)[2]
            wn_key = find_wn_key(sentence, lookup_word)
            results_file.write("|" + linkup_key + "|     " + wn_key + "\n")
            if linkup_key in annotation_results:
                total_sentence_count += 1
                wn_keylist = annotation_results[linkup_key]
                if len(wn_keylist) > 0:
                    most_frequent_wn_key = max(set(wn_keylist), key=wn_keylist.count)
                    if most_frequent_wn_key == wn_key:
                        print("correct wsd for", linkup_key, wn_key)
                        output_file.write("correct wsd for " + linkup_key + " " + wn_key + "\n")
                        correct_count += 1
                        print("correct", correct_count, "| total", total_sentence_count)
                        output_file.write("correct " + str(correct_count) + " | total " + str(total_sentence_count) + "\n")
                    else:
                        print("incorrect wsd for", linkup_key, "| found", wn_key, ", correct is", most_frequent_wn_key)
                        output_file.write("incorrect wsd for " + linkup_key + " | found " + wn_key + ", correct is " + most_frequent_wn_key + "\n")
            else:
                invalid_linkup_key_count += 1
                print("linkup key", linkup_key, "not found in gold standard clean dataset")
                output_file.write("linkup key " + linkup_key + " not found in gold standard clean dataset\n")

results_file.close()
print("total invalid linkup key count", invalid_linkup_key_count)
output_file.write("total invalid linkup key count " + str(invalid_linkup_key_count) + "\n")
print("total correct", correct_count)
output_file.write("total correct " + str(correct_count) + "\n")
print("total sentences", total_sentence_count)
output_file.write("total sentences " + str(total_sentence_count) + "\n")
output_file.close()
