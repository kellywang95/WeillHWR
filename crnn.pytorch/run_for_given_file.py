import refactored_main

from spell import correction

import numpy as np

# Get Opt
opt = refactored_main.get_parameters()

# Init CRNN
crnn, converter, _ = refactored_main.load_trained_crnn_for_eval(opt)

# Image mapping
image_mapping = {'1': 'a01/a01-003u/', '2': 'a01/a01-003/'}


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
    extract_result("2")