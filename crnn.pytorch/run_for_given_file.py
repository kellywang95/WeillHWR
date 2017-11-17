import refactored_main, operator

from spell import correction

import numpy as np

from glob import glob

from collections import defaultdict
import functools
# Get Opt
opt = refactored_main.get_parameters()

# Init CRNN
crnn, converter, _ = refactored_main.load_trained_crnn_for_eval(opt)

# Image mapping
image_mapping = {'1': 'a01/a01-003u/', '2': 'a01/a01-003/'}

# curate documents
doc_to_text = defaultdict(int)
for file in glob('data/med_ground_truth/*.txt'):
    wordcount = defaultdict(int)
    for word in open(file).read().lower().split():
        wordcount[word] += 1

    doc_to_text[file] = wordcount


def get_most_relevant(keyword_string):
    keys = keyword_string.split(",")

    score_list = []
    for file in doc_to_text:
        score_list.append((functools.reduce(operator.mul, [doc_to_text[file][keyword.strip()] for keyword in keys], 1), file))

    most_frequent = sorted(score_list, key=lambda x: x[0], reverse=True)[0]

    # Todo preprocessing on top of the test sent back?
    return {'name': most_frequent[1].split("/")[-1].rstrip(" copy.txt"), 'text' : open(most_frequent[1]).read()} if most_frequent[0] != 0 else None, None


def extract_result(image_index):

    # get the Predicted List
    predicted_list = refactored_main.extract_result(opt, crnn, converter, image_mapping[image_index])

    match = np.array([1 if prediction.target.lower() == prediction.pred.lower() else 0 for prediction in predicted_list])
    print("Model Prediction : ")
    print(" ".join([prediction.pred.lower() for prediction in predicted_list]))
    print('Accuracy : ' + str(match.mean()))

    corrected_text = [correction(prediction.pred.lower()) for prediction in predicted_list]

    # Optimize this.

    corrected_match = np.array([1 if corrected.lower() == prediction.target.lower() else 0 for corrected, prediction in zip(corrected_text, predicted_list)])
    print("Corrected Prediction : ")
    print(" ".join(corrected_text))
    print('Accuracy : ' + str(corrected_match.mean()))

    # we also need to calculate the accuracy for entries only larger than or equal to 5 terms

    return " ".join(corrected_text)


if __name__ == '__main__':
    print(get_most_relevant('wound, spinal, disease'))