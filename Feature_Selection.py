"""
Identify features with zero importance according to Random Forest model,
then delete those features from original contents dataframe
Version: 29.10.2023, 17:30(UTC+9)
"""


def identify_zero_importance_feature(feature_importance_dataframe_param):
    """
    Identy features with zero importance, according to the result of Random Forest
    :param feature_importance_dataframe_param: The sorted feature importance in dataframe
    :return: A list containing feature names with zero importance score
    """
    # Get the index of the first row of data with an importance value of 0
    first_zero_importance_feature_index = (feature_importance_dataframe_param["Importance"] == 0).idxmax()

    # Extract zero importance features to an individual dataframe
    zero_importance_feature_name_dataframe = feature_importance_dataframe_param.iloc[
                                             first_zero_importance_feature_index:]

    # Convert dataframe to list format
    zero_importance_feature_name_list = zero_importance_feature_name_dataframe.iloc[:, 0].values.tolist()

    print("There are", len(zero_importance_feature_name_list), "features with zero importance. They would be deleted")

    return zero_importance_feature_name_list


def feature_selection(zero_importance_feature_name_list_param, original_contents_dataframe_with_all_features_param):
    """
    Delete those features with zero importance from original dataframe,
     containing label and data contents reading from CSV files
    :param zero_importance_feature_name_list_param: The list containing feature names with zero importance score
    :param original_contents_dataframe_with_all_features_param: The original dataframe reading from CSV files,
     containing both label and data contents
    :return: A dataframe containing label and data contents with filtered features
    """
    # Delete zero importance features from original contents dataframe
    filtered_features_data_and_label_dataframe = original_contents_dataframe_with_all_features_param.drop(
        zero_importance_feature_name_list_param, axis=1)

    print("The shape of dataframe after feature selection:", filtered_features_data_and_label_dataframe.shape)

    return filtered_features_data_and_label_dataframe


if __name__ == '__main__':
    # Get the feature importance dataframe based on training data

    import Read_Source_Data

    (return_train_data_concatenated_dataframe, return_train_data_data_dataframe,
     return_train_data_text_label_dataframe) = Read_Source_Data.training_data_from_csv_file(
        training_data_from_a_param=Read_Source_Data.training_data_a,
        training_data_from_c_param=Read_Source_Data.training_data_c,
        proportion_param=1)

    import Generate_Dataset

    return_encoded_label_data_dataframe, return_text_label_list = Generate_Dataset.create_dataset_train(
        entire_dataset_param=return_train_data_concatenated_dataframe)

    return_x_train, return_x_test, return_y_train, return_y_test = Generate_Dataset.split_dataset_into_train_and_test(
        entire_dataset_param=return_encoded_label_data_dataframe)

    import Random_Forest

    return_model, return_feature_importance = Random_Forest.train_random_forest(
        model_param=Random_Forest.random_forest_params,
        x_train_param=return_x_train, y_train_param=return_y_train)

    # Feature selection
    return_zero_importance_feature_name_list = identify_zero_importance_feature(
        feature_importance_dataframe_param=return_feature_importance)

    return_filtered_features_data_and_label_dataframe = feature_selection(
        zero_importance_feature_name_list_param=return_zero_importance_feature_name_list,
        original_contents_dataframe_with_all_features_param=return_train_data_concatenated_dataframe)

    print(return_filtered_features_data_and_label_dataframe, return_filtered_features_data_and_label_dataframe.shape)

    # Test-c
    (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
     return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
        test_data_csv_file_path_param=Read_Source_Data.test_data_c)

    import Generate_Dataset

    return_test_data_dataframe, return_test_encoded_label_dataframe, return_test_text_label_list = (
        Generate_Dataset.create_dataset_test(
            test_dataset_dataframe_param=return_test_data_csv_file_contents_dataframe))

    # Feature selection
    return_zero_importance_feature_name_list = identify_zero_importance_feature(
        feature_importance_dataframe_param=return_feature_importance)

    return_filtered_features_data_and_label_dataframe = feature_selection(
        zero_importance_feature_name_list_param=return_zero_importance_feature_name_list,
        original_contents_dataframe_with_all_features_param=return_test_data_csv_file_contents_dataframe)

    print(return_filtered_features_data_and_label_dataframe, return_filtered_features_data_and_label_dataframe.shape)
