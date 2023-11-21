"""
Combination of transformer model and CNN model
Version: 31.10.2023, 22:45(UTC+9)
"""

import tensorflow as tf


class Transformer(tf.keras.models.Model):
    # Transformer model
    def __init__(self):
        super(Transformer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=64)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=32)])

        self.normalization_1 = tf.keras.layers.LayerNormalization(epsilon=0.000001)
        self.normalization_2 = tf.keras.layers.LayerNormalization(epsilon=0.000001)

        self.dropout_1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.1)

    def call(self, inputs):
        attention_output = self.mha(inputs, inputs, inputs)
        output_1 = self.dropout_1(attention_output)
        output_1 = self.normalization_1(inputs + output_1)

        ffn_output = self.ffn(output_1)
        output_2 = self.dropout_2(ffn_output)
        result = self.normalization_2(output_1 + output_2)

        return result


class CNNTransformerModel(tf.keras.models.Model):
    # 1D CNN model
    def __init__(self, num_classes):
        super(CNNTransformerModel, self).__init__()

        self.conv_f1_k1 = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding="same", activation="relu")
        self.conv_f2_k2 = tf.keras.layers.Conv1D(filters=2, kernel_size=2, strides=1, padding="same", activation="relu")
        self.conv_f4_k4 = tf.keras.layers.Conv1D(filters=4, kernel_size=4, strides=1, padding="same", activation="relu")

        self.conv_f16_k1 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding="same",
                                                  activation="relu")
        self.conv_f32_k2 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=1, padding="same",
                                                  activation="relu")
        self.conv_f64_k4 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=1, padding="same",
                                                  activation="relu")

        self.normalization_1 = tf.keras.layers.Normalization()

        self.max_pooling = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.transformer_layer = Transformer()
        self.avg_pooling = tf.keras.layers.AveragePooling1D(pool_size=2)

        self.conv_f128_k3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=10, padding="same",
                                                   activation="relu")
        self.conv_f256_k3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=10, padding="same",
                                                   activation="relu")
        self.normalization_2 = tf.keras.layers.Normalization()

        self.flatten_layer = tf.keras.layers.Flatten()

        self.dense_64 = tf.keras.layers.Dense(units=64, activation="relu")
        self.dense_32 = tf.keras.layers.Dense(units=32, activation="relu")

        self.dropout = tf.keras.layers.Dropout(rate=0.1)

        self.output_layer = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs):
        cnn_output = self.normalization_1(inputs)

        cnn_output = self.conv_f1_k1(cnn_output)
        cnn_output = self.conv_f2_k2(cnn_output)
        cnn_output = self.conv_f4_k4(cnn_output)

        cnn_output = self.normalization_1(cnn_output)

        cnn_output = self.conv_f16_k1(cnn_output)
        cnn_output = self.conv_f32_k2(cnn_output)
        cnn_output = self.conv_f64_k4(cnn_output)

        cnn_output = self.max_pooling(cnn_output)

        trans_output = self.transformer_layer(inputs)
        trans_output = self.avg_pooling(trans_output)

        x = tf.concat([0.4 * cnn_output, 0.6 * trans_output], axis=-1)
        x = self.normalization_2(x)

        x = self.conv_f128_k3(x)
        x = self.conv_f256_k3(x)

        x = self.flatten_layer(x)

        x = self.dense_64(x)
        x = self.dense_32(x)
        x = self.dropout(x)
        result = self.output_layer(x)

        return result


def train_cnn_model(x_train_param, y_train_param, epochs_param, batch_param):
    # Initialize model
    cnn_model = CNNTransformerModel(num_classes=16)

    # Compile model
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=tf.keras.metrics.sparse_categorical_accuracy)

    # Show model structure
    cnn_model.build(input_shape=(None, single_item_length, 1))
    cnn_model.summary()

    model.fit(x=x_train_param, y=y_train_param, epochs=epochs_param, batch_size=batch_param)

    return cnn_model


if __name__ == '__main__':
    # Feature number
    single_item_length = 1859
    model = CNNTransformerModel(num_classes=16)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=tf.keras.metrics.sparse_categorical_accuracy)
    model.build(input_shape=(None, single_item_length, 1))
    model.summary()

    # Start
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

    return_filtered_x_train = return_filtered_x_train.values.reshape(return_filtered_x_train.shape[0],
                                                                     return_filtered_x_train.shape[1], 1)

    print(return_filtered_x_train.shape)
    print(return_filtered_y_train.shape)

    model = train_cnn_model(return_filtered_x_train, return_filtered_y_train, epochs_param=2, batch_param=8)

    # +++++++++++++++++++++++
    # Test on test data from training data
    print("Test on training data")

    return_filtered_x_test = return_filtered_x_test.values.reshape(return_filtered_x_test.shape[0],
                                                                   return_filtered_x_test.shape[1], 1)

    y_predict = model.predict(x=return_filtered_x_test)
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
    return_filtered_test_data_dataframe = return_filtered_test_data_dataframe.values.reshape(
        return_filtered_test_data_dataframe.shape[0],
        return_filtered_test_data_dataframe.shape[1], 1)
    y_predict = model.predict(X=return_filtered_test_data_dataframe)

    from sklearn import metrics

    print(metrics.accuracy_score(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))

    print(metrics.classification_report(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))

    print(metrics.confusion_matrix(y_true=return_filtered_test_encoded_label_dataframe, y_pred=y_predict))
