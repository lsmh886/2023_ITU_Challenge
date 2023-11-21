"""
Read original data from csv files, combine training data from two domains
Version: 28.10.2023, 19:45(UTC+9)
"""

import pandas as pd

training_data_a = "/home/nakaolab/Desktop/challenge/data/training-data_a.csv"
training_data_c = "/home/nakaolab/Desktop/challenge/data/training-data_c.csv"
test_data_c = "/home/nakaolab/Desktop/challenge/data/test-data_c.csv"


def training_data_from_csv_file(training_data_from_a_param, training_data_from_c_param, proportion_param):
    """
    Read training data from two csv files: training_data_a and training_data_c, then, concatenate into one.
    The proportion of data come from training_data_c is various
    :param training_data_from_a_param: The absolute path fo training_data_a
    :param training_data_from_c_param: The absolute path fo training_data_c
    :param proportion_param: The proportion of data come from training_data_c
    :return:
      train_data_concatenated_dataframe: Dataframe containing the training data from two csv files
      train_data_data_dataframe: Dataframe containing metrics (data)
      train_data_text_label_dataframe: Dataframe containing label in text format
    """

    # Read from CSV file
    training_a_csv_file_contents_dataframe = pd.read_csv(filepath_or_buffer=training_data_from_a_param)
    training_c_csv_file_contents_dataframe = pd.read_csv(filepath_or_buffer=training_data_from_c_param)

    # Shuffle the data
    training_a_csv_file_contents_dataframe = training_a_csv_file_contents_dataframe.sample(frac=1, random_state=10)
    training_c_csv_file_contents_dataframe = training_c_csv_file_contents_dataframe.sample(frac=1, random_state=10)

    training_a_csv_file_contents_dataframe = training_a_csv_file_contents_dataframe.sample(frac=1, random_state=20)
    training_c_csv_file_contents_dataframe = training_c_csv_file_contents_dataframe.sample(frac=1, random_state=20)

    training_a_csv_file_contents_dataframe = training_a_csv_file_contents_dataframe.sample(frac=1, random_state=30)
    training_c_csv_file_contents_dataframe = training_c_csv_file_contents_dataframe.sample(frac=1, random_state=30)

    # Get a certain number of rows of data proportionally from training_data_c
    sampled_training_c_csv_file_contents_dataframe = training_c_csv_file_contents_dataframe.sample(
        frac=proportion_param, random_state=42)

    print("Training A and C shape(Original, without feature selection):", training_a_csv_file_contents_dataframe.shape,
          training_c_csv_file_contents_dataframe.shape)

    # Concat training data a and sampled training data c
    train_data_concatenated_dataframe = pd.concat(
        [training_a_csv_file_contents_dataframe, sampled_training_c_csv_file_contents_dataframe])

    # Data columns
    train_data_data_dataframe = train_data_concatenated_dataframe.iloc[:, 3:]

    # Label columns in text format
    train_data_text_label_dataframe = pd.DataFrame(data=train_data_concatenated_dataframe["y_true(fc)"])

    return train_data_concatenated_dataframe, train_data_data_dataframe, train_data_text_label_dataframe


def test_data_from_csv_file(test_data_csv_file_path_param):
    """
    Read data from CSV file containing test data into dataframe
    :param test_data_csv_file_path_param: The absolute path of CSV file containing test data
    :return: Three dataframes
      test_data_csv_file_contents_dataframe: Dataframe containing the entire CSV contents
      test_data_training_data_dataframe: Dataframe containing metrics (data)
      test_data_text_label_dataframe: Dataframe containing label in text format
    """

    # Read test data from CSV file
    test_data_csv_file_contents_dataframe = pd.read_csv(filepath_or_buffer=test_data_csv_file_path_param)

    # Shuffle the data
    test_data_csv_file_contents_dataframe = test_data_csv_file_contents_dataframe.sample(frac=1, random_state=10)
    test_data_csv_file_contents_dataframe = test_data_csv_file_contents_dataframe.sample(frac=1, random_state=10)

    # Data columns
    test_data_data_dataframe = test_data_csv_file_contents_dataframe.iloc[:, 3:]

    # Label column in text format
    test_data_text_label_dataframe = pd.DataFrame(data=test_data_csv_file_contents_dataframe["y_true(fc)"])

    print("Test C shape(Original, without feature selection):", test_data_csv_file_contents_dataframe.shape)

    return test_data_csv_file_contents_dataframe, test_data_data_dataframe, test_data_text_label_dataframe


if __name__ == '__main__':
    (return_train_data_concatenated_dataframe, return_train_data_data_dataframe,
     return_train_data_text_label_dataframe) = training_data_from_csv_file(
        training_data_from_a_param=training_data_a, training_data_from_c_param=training_data_c,
        proportion_param=1)

    (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
     return_test_data_text_label_dataframe) = test_data_from_csv_file(
        test_data_csv_file_path_param=test_data_c)

    print(return_train_data_concatenated_dataframe.shape, return_train_data_data_dataframe.shape,
          return_train_data_text_label_dataframe.shape)
    print(return_test_data_csv_file_contents_dataframe.shape, return_test_data_data_dataframe.shape,
          return_test_data_text_label_dataframe.shape)
