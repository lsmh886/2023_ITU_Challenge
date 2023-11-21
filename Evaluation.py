"""
Evaluate the trained model
Version: 29.10.2023, 17:45(UTC+9)
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def model_evaluation(model_param, x_test_param, y_test_param, text_label_list_param):
    """
    Evaluate the trained model on test dataset
    :param model_param: The trained model
    :param x_test_param: Test data
    :param y_test_param: Test label
    :param text_label_list_param: The list containing text label names
    :return: Model accuracy, confusion matrix, and classification report
    """

    # Prediction result from trained model
    y_model_predict = model_param.predict(X=x_test_param)

    # Model accuracy
    model_accuracy = metrics.accuracy_score(y_true=y_test_param, y_pred=y_model_predict)

    # Classification report
    classification_report = metrics.classification_report(y_true=y_test_param, y_pred=y_model_predict,
                                                          target_names=text_label_list_param)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true=y_test_param, y_pred=y_model_predict)

    print("+++" * 10)
    print("Accuracy:", model_accuracy)
    print("---" * 10)

    print("Confusion Matrix:\n", confusion_matrix)
    print("---" * 10)

    print("Classification Report:\n", classification_report)
    print("+++" * 10)

    plt.figure(figsize=(10, 10))
    sns.heatmap(data=confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar="False", square=True,
                xticklabels=text_label_list_param, yticklabels=text_label_list_param)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()

    return model_accuracy, confusion_matrix, classification_report


if __name__ == '__main__':
    # # No feature selection
    # # Training-a and training-c
    # import Read_Source_Data

    # (return_train_data_concatenated_dataframe, return_train_data_data_dataframe,
    #  return_train_data_text_label_dataframe) = Read_Source_Data.training_data_from_csv_file(
    #     training_data_from_a_param=Read_Source_Data.training_data_a,
    #     training_data_from_c_param=Read_Source_Data.training_data_c,
    #     proportion_param=1)

    # import Generate_Dataset

    # return_encoded_label_data_dataframe, return_text_label_list = Generate_Dataset.create_dataset_train(
    #     entire_dataset_param=return_train_data_concatenated_dataframe)

    # return_x_train, return_x_test, return_y_train, return_y_test = Generate_Dataset.split_dataset_into_train_and_test(
    #     entire_dataset_param=return_encoded_label_data_dataframe)

    # import Random_Forest
    # return_model, return_feature_importance = Random_Forest.train_random_forest(
    #     model_param=Random_Forest.random_forest_params,
    #     x_train_param=return_x_train,
    #     y_train_param=return_y_train)

    # # Test-c
    # (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
    #  return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
    #     test_data_csv_file_path_param=Read_Source_Data.test_data_c)

    # return_test_data_dataframe, return_test_encoded_label_dataframe, return_test_text_label_list = (
    #     Generate_Dataset.create_dataset_test(
    #         test_dataset_dataframe_param=return_test_data_csv_file_contents_dataframe))

    # # Test on test data from training data
    # print("Test on training data")
    # model_evaluation(model_param=return_model, x_test_param=return_x_test, y_test_param=return_y_test,
    #                 text_label_list_param=return_text_label_list)

    # # +++++++++++++++++++++++
    # # Test on test-c data
    # print("Test on test data")
    # model_evaluation(model_param=return_model, x_test_param=return_test_data_dataframe,
    #                  y_test_param=return_test_encoded_label_dataframe,
    #                  text_label_list_param=return_test_text_label_list)

    # print("Feature importance:\n", return_x_test.shape, return_test_data_dataframe.shape)

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

    import Random_Forest

    return_model, return_feature_importance = Random_Forest.train_random_forest(
        model_param=Random_Forest.random_forest_params,
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

    import Random_Forest

    return_filtered_model, return_filtered_feature_importance = Random_Forest.train_random_forest(
        model_param=Random_Forest.random_forest_params,
        x_train_param=return_filtered_x_train, y_train_param=return_filtered_y_train)

    # +++++++++++++++++++++++
    # Test on test data from training data
    print("Test on training data")
    model_evaluation(model_param=return_filtered_model, x_test_param=return_filtered_x_test,
                     y_test_param=return_filtered_y_test,
                     text_label_list_param=return_filtered_text_label_list)

    # Test-c
    (return_test_data_csv_file_contents_dataframe, return_test_data_data_dataframe,
     return_test_data_text_label_dataframe) = Read_Source_Data.test_data_from_csv_file(
        test_data_csv_file_path_param=Read_Source_Data.test_data_c)

    import Feature_Selection

    return_filtered_features_data_and_label_dataframe = Feature_Selection.feature_selection(
        zero_importance_feature_name_list_param=return_zero_importance_feature_name_list,
        original_contents_dataframe_with_all_features_param=return_test_data_csv_file_contents_dataframe)

    import Generate_Dataset

    (return_filtered_test_data_dataframe, return_filtered_test_encoded_label_dataframe,
     return_filtered_test_text_label_list) = (
        Generate_Dataset.create_dataset_test(
            test_dataset_dataframe_param=return_filtered_features_data_and_label_dataframe))

    # Test on test-c data
    print("Test on test data")
    model_evaluation(model_param=return_filtered_model, x_test_param=return_filtered_test_data_dataframe,
                     y_test_param=return_filtered_test_encoded_label_dataframe,
                     text_label_list_param=return_filtered_test_text_label_list)

    print("Data shape:\n", return_x_test.shape, return_filtered_test_data_dataframe.shape)
