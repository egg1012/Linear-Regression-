import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

def trim(dataset, glove):
    data= load_tsv_dataset(dataset)
    gloveMap= load_feature_dictionary(glove)
    array=[]
    for review in data:
        len=0
        tmp= np.zeros(300)
        for word in review[1].split(' '):
            if word in gloveMap:
                len+=1
                tmp= np.add(tmp,gloveMap[word])
        tmp= tmp*(1/len)
        array.append((review[0], tmp))
    return array 

trainthing= trim(args.train_input, args.feature_dictionary_in)
valthing = trim(args.validation_input, args.feature_dictionary_in)
testthing= trim(args.test_input, args.feature_dictionary_in)

with open(args.train_out, "w") as f_out:
    for item in trainthing:
        thing= '{:.6f}'.format(item[0])
        f_out.write(thing)
        for char in item[1]:
            thing= '{:.6f}'.format(char)
            f_out.write("\t"+thing)
        f_out.write("\n")

with open(args.validation_out, "w") as f_out:
    for item in valthing:
        thing= '{:.6f}'.format(item[0])
        f_out.write(thing+"\t")
        for char in item[1]:
            thing= '{:.6f}'.format(char)
            f_out.write(thing+"\t")
        f_out.write("\n")

with open(args.test_out, "w") as f_out:
    for item in testthing:
        thing= '{:.6f}'.format(item[0])
        f_out.write(thing+"\t")
        for char in item[1]:
            thing= '{:.6f}'.format(char)
            f_out.write(thing+"\t")
        f_out.write("\n")
    