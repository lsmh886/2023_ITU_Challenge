"""
Encode label, generate train dataset and test dataset model training
Version: 29.10.2023, 00:15(UTC+9)
"""

import pandas as pd

from sklearn import preprocessing, model_selection


def encode_label(text_label_series_param):
    """
    Encode text label
    :param text_label_series_param: The series containing labels
    :return: 
      encoded_label_series: Encoded label in dataframe format
      text_label_list: The list containing label names
    """
    # Instantiate label encoder
    label_encoder = preprocessing.LabelEncoder()

    print("Label shape:", text_label_series_param.shape)

    # Text label list
    text_label_list = text_label_series_param.unique().tolist()

    # Encode the text label
    encoded_label_series = label_encoder.fit_transform(text_label_series_param)

    return encoded_label_series, text_label_list


def create_dataset_train(entire_dataset_param):
    """
    Combine and create dataset containing both labels and training data
    :param entire_dataset_param: The dataframe containing un-encoded labels and training data,
      including label and data come from different sources
    :return: 
      encoded_label_data_dataframe: A dataframe including all encoded label and training data
      text_label_list: The list containing label names
    """
    # Encode label
    encoded_label_series, text_label_list = encode_label(text_label_series_param=entire_dataset_param.iloc[:, 2])
    encoded_label_dataframe = pd.DataFrame(data=encoded_label_series, columns=["encoded_label"]).reset_index(drop=True)

    # Split label and data
    data_dataframe = entire_dataset_param.iloc[:, 3:].reset_index(drop=True)

    # Combine label and data, encoded label dataframe is the first column after concatenating
    encoded_label_data_dataframe = pd.concat([encoded_label_dataframe, data_dataframe], axis=1)

    print("Encode label and data shape:", encoded_label_data_dataframe.shape)

    return encoded_label_data_dataframe, text_label_list


def split_dataset_into_train_and_test(entire_dataset_param):
    """
    Split the entire dataset into training dataset and test dataset
    :param entire_dataset_param: The dataframe containing all encoded label and training data
    :return: 4 datasets: x_train, x_test, y_train, y_test
    """
    # Y_label
    y_label = entire_dataset_param.iloc[:, 0]

    # X_data
    x_data = entire_dataset_param.iloc[:, 1:]

    # Split to train dataset and test dataset
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_label,
                                                                        test_size=0.2, random_state=10)

    return x_train, x_test, y_train, y_test


def create_dataset_test(test_dataset_dataframe_param):
    # Instantiate label encoder
    label_encoder = preprocessing.LabelEncoder()

    # Encode the text label
    test_encoded_label_series = label_encoder.fit_transform(test_dataset_dataframe_param.iloc[:, 2])
    test_encoded_label_dataframe = pd.DataFrame(data=test_encoded_label_series, columns=["encoded_label"]).reset_index(
        drop=True)

    # Text label list
    test_text_label_list = test_dataset_dataframe_param.iloc[:, 2].unique().tolist()

    # Test data dataframe
    test_data_dataframe = test_dataset_dataframe_param.iloc[:, 3:]

    return test_data_dataframe, test_encoded_label_dataframe, test_text_label_list


if __name__ == '__main__':
    import Read_Source_Data

    # Training-a and training-c
    (return_train_data_concatenated_dataframe, return_train_data_data_dataframe,
     return_train_data_text_label_dataframe) = Read_Source_Data.training_data_from_csv_file(
        training_data_from_a_param=Read_Source_Data.training_data_a,
        training_data_from_c_param=Read_Source_Data.training_data_c,
        proportion_param=1)

    return_encoded_label_data_dataframe, return_text_label_list = create_dataset_train(
        entire_dataset_param=return_train_data_concatenated_dataframe)

    return_x_train, return_x_test, return_y_train, return_y_test = split_dataset_into_train_and_test(
        entire_dataset_param=return_encoded_label_data_dataframe)

    # Test-c
    (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
     return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
        test_data_csv_file_path_param=Read_Source_Data.test_data_c)

    return_test_data_dataframe, return_test_encoded_label_dataframe, return_test_text_label_list = create_dataset_test(
        test_dataset_dataframe_param=return_test_data_csv_file_contents_dataframe)

    print(return_test_data_dataframe, return_test_encoded_label_dataframe, return_test_text_label_list)
