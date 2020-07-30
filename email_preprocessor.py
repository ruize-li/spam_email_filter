'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Ruize Li
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import operator
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron', stopwords = False):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Update the counts of each words in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    - When reading in email files, you might experience errors due to reading funky characters
    (spam can contain weird things!). On this dataset, this can be fixed by telling Python to assume
    each file is encoded using 'latin-1': encoding='latin-1'
    '''
    # read stop Words
    with open('stopwords', 'r', encoding = 'latin-1') as sw:
        sw_list = tokenize_words(sw.read())
        # print(sw_list)
    # read files
    list_files = list()
    for (dirpath, dirnames, filenames) in os.walk(email_path):
        list_files += [os.path.join(dirpath, file) for file in filenames]
    # read file as string and update dictionary
    word_freq = {}
    for f in list_files:
        with open(f, 'r', encoding='latin-1') as file:
            word_list = tokenize_words(file.read())
            if stopwords:
                for word in word_list:
                    if word not in sw_list:
                        word_freq[word] = word_freq.get(word, 0) + 1
            else:
                for word in word_list:
                    word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq, len(list_files)-1
    pass


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    sorted_word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse = True)}
    # print(list(sorted_word_freq.keys()))

    return list(sorted_word_freq.keys())[:num_features], list(sorted_word_freq.values())[:num_features]
    pass


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    # read files
    list_files = list()
    for (dirpath, dirnames, filenames) in os.walk(email_path):
        if dirpath == 'data/enron':
            continue
        for file in filenames:
            list_files += [os.path.join(dirpath, file)]
        # print(dirpath)

    # print(dirpath)
    # read file as string and update dictionary
    feats = np.zeros((num_emails, len(top_words)))
    y = np.zeros((num_emails,))
    i = 0
    # print(list_files)
    for f in list_files:
        # print(word_freq.keys())
        with open(f, 'r', encoding='latin-1') as file:
            if not os.path.basename(f).startswith('.'):
                word_freq = {word : 0 for word in top_words}
                # set y
                if 'ham' in os.path.dirname(f):
                    y[i] = 0
                elif 'spam' in os.path.dirname(f) :
                    # print('spam detected', i)
                    y[i] = 1
                word_list = tokenize_words(file.read())
                for word in word_list:
                    if word in top_words:
                        word_freq[word] += 1
                    else:
                        pass
                # add to feature vector
                feats[i] = np.array(list(word_freq.values()))
                i+=1
    # print(feats[0])
    return feats, y
    pass


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].

    HINTS:
    - If you're shuffling, work with indices rather than actual values.
    '''
    # shuffle if needed
    index = np.arange(len(features))
    if shuffle:
        np.random.shuffle(index)

    # divide dataset by proportions
    split_pt = int(test_prop * len(features))

    test_idx = index[:split_pt]
    train_idx = index[split_pt:]

    x_test = features[test_idx]
    x_train = features[train_idx]
    y_test = y[test_idx]
    y_train = y[train_idx]
    return x_train, y_train, train_idx, x_test, y_test, test_idx

    pass

def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''


    str = []
    # read files
    list_files = list()
    for (dirpath, dirnames, filenames) in os.walk(email_path):
        if dirpath == 'data/enron':
            continue
        for file in filenames:
            list_files += [os.path.join(dirpath, file)]
    # print(list_files)
    for i in range(len(inds)):

        with open(list_files[inds[i]], 'r', encoding = 'latin-1') as file:
            # print(file)
            # print('openning: ', inds[i])
            str.append(file.read())
    return str
    pass
