"""
Train a Random Forest model, and calculate the feature importance
Version: 29.10.2023, 17:15(UTC+9)
"""

import pandas as pd

from sklearn import ensemble

random_forest_params = {"n_estimators": 200,
                        "max_depth": 100,
                        "random_state": 20}


def train_random_forest(model_param, x_train_param, y_train_param):
    """
    Train a Random Forest model, and get the sorted feature importance
    :param model_param: Model initialization parameters
    :param x_train_param: Training data
    :param y_train_param: Training label
    :return: 
      random_forest_model: A trained Random Forest model
      sorted_feature_importance_dataframe: The sorted feature importance dataframe
    """
    # Initialize random forest model
    random_forest_model = ensemble.RandomForestClassifier(**model_param)

    # Train the model
    random_forest_model.fit(X=x_train_param, y=y_train_param)

    # Get feature importance
    feature_importance = random_forest_model.feature_importances_

    # Feature names
    feature_name = x_train_param.columns

    # Feature importance dataframe
    feature_importance_dataframe = pd.DataFrame({"Features": feature_name, "Importance": feature_importance})

    # Feature importance sorting
    sorted_feature_importance_dataframe = feature_importance_dataframe.sort_values(by="Importance",
                                                                                   ascending=False).reset_index(
        drop=True)

    return random_forest_model, sorted_feature_importance_dataframe


if __name__ == '__main__':
    # # No feature selection
    # # Training-a and training-c
    # import Read_Source_Data
    #
    # (return_train_data_concatenated_dataframe, return_train_data_data_dataframe,
    #  return_train_data_text_label_dataframe) = Read_Source_Data.training_data_from_csv_file(
    #     training_data_from_a_param=Read_Source_Data.training_data_a,
    #     training_data_from_c_param=Read_Source_Data.training_data_c,
    #     proportion_param=1)
    #
    # import Generate_Dataset
    #
    # return_encoded_label_data_dataframe, return_text_label_list = Generate_Dataset.create_dataset_train(
    #     entire_dataset_param=return_train_data_concatenated_dataframe)
    #
    # return_x_train, return_x_test, return_y_train, return_y_test = Generate_Dataset.split_dataset_into_train_and_test(
    #     entire_dataset_param=return_encoded_label_data_dataframe)
    #
    # # Test-c
    # (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
    #  return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
    #     test_data_csv_file_path_param=Read_Source_Data.test_data_c)
    #
    # return_test_data_dataframe, return_test_encoded_label_dataframe, return_test_text_label_list = (
    #     Generate_Dataset.create_dataset_test(
    #         test_dataset_dataframe_param=return_test_data_csv_file_contents_dataframe))
    #
    # return_model, return_feature_importance = train_random_forest(model_param=random_forest_params,
    #                                                               x_train_param=return_x_train,
    #                                                               y_train_param=return_y_train)
    #
    # # # Test on test data from training data
    # # y_predict = return_model.predict(X=return_x_test)
    #
    # # from sklearn import metrics
    #
    # # print(metrics.accuracy_score(y_true=return_y_test, y_pred=y_predict))
    #
    # # print(metrics.classification_report(y_true=return_y_test, y_pred=y_predict))
    #
    # # print(metrics.confusion_matrix(y_true=return_y_test, y_pred=y_predict))
    #
    # # +++++++++++++++++++++++
    # # Test on test-c data
    # y_predict = return_model.predict(X=return_test_data_dataframe)
    #
    # from sklearn import metrics
    #
    # print(metrics.accuracy_score(y_true=return_test_encoded_label_dataframe, y_pred=y_predict))
    #
    # print(metrics.classification_report(y_true=return_test_encoded_label_dataframe, y_pred=y_predict))
    #
    # print(metrics.confusion_matrix(y_true=return_test_encoded_label_dataframe, y_pred=y_predict))
    #
    # print("Feature importance:\n", return_feature_importance)

    # ------------------------------------------------
    # With feature selection
    # Training-a and training-c
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

    return_model, return_feature_importance = train_random_forest(
        model_param=random_forest_params,
        x_train_param=return_x_train, y_train_param=return_y_train)

    import Feature_Selection

    return_zero_importance_feature_name_list = Feature_Selection.identify_zero_importance_feature(
        feature_importance_dataframe_param=return_feature_importance)

    return_filtered_features_data_and_label_dataframe = Feature_Selection.feature_selection(
        zero_importance_feature_name_list_param=return_zero_importance_feature_name_list,
        original_contents_dataframe_with_all_features_param=return_train_data_concatenated_dataframe)

    import Generate_Dataset

    return_filtered_encoded_label_data_dataframe, return_filtered_text_label_list = (
        Generate_Dataset.create_dataset_train(
            entire_dataset_param=return_filtered_features_data_and_label_dataframe))

    return_filtered_x_train, return_filtered_x_test, return_filtered_y_train, return_filtered_y_test = (
        Generate_Dataset.split_dataset_into_train_and_test(
            entire_dataset_param=return_filtered_encoded_label_data_dataframe))

    return_filtered_model, return_filtered_feature_importance = train_random_forest(
        model_param=random_forest_params,
        x_train_param=return_filtered_x_train, y_train_param=return_filtered_y_train)

    # +++++++++++++++++++++++
    # Test on test data from training data
    print("Test on training data")
    y_predict = return_filtered_model.predict(X=return_filtered_x_test)
    print(return_filtered_x_test.shape)

    from sklearn import metrics

    print(metrics.accuracy_score(y_true=return_filtered_y_test, y_pred=y_predict))

    print(metrics.classification_report(y_true=return_filtered_y_test, y_pred=y_predict))

    print(metrics.confusion_matrix(y_true=return_filtered_y_test, y_pred=y_predict))

    # Test-c

    (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
     return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
        test_data_csv_file_path_param=Read_Source_Data.test_data_c)

    import Feature_Selection

    return_filtered_features_data_and_label_dataframe = Feature_Selection.feature_selection(
        zero_importance_feature_name_list_param=return_zero_importance_feature_name_list,
        original_contents_dataframe_with_all_features_param=return_test_data_csv_file_contents_dataframe)

    import Generate_Dataset

    return_filtered_test_data_dataframe, return_filtered_test_encoded_label_dataframe, return_test_text_label_list = (
        Generate_Dataset.create_dataset_test(
            test_dataset_dataframe_param=return_filtered_features_data_and_label_dataframe))

    # Test on test-c data
    print("Test on test data")
    y_predict = return_filtered_model.predict(X=return_filtered_test_data_dataframe)

    from sklearn import metrics

    print(metrics.accuracy_score(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))

    print(metrics.classification_report(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))

    print(metrics.confusion_matrix(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))
