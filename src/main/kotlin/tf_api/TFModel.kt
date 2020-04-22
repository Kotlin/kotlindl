package tf_api

import tensorflow.training.util.ImageDataset
import tf_api.inference.InferenceTFModel
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.Optimizer

abstract class TFModel<T : Number> : InferenceTFModel() {

    /**
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    abstract fun compile(
        optimizer: Optimizer<T>,
        loss: LossFunctions = LossFunctions.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        metric: Metrics = Metrics.ACCURACY
    )

    abstract fun fit(trainDataset: ImageDataset, epochs: Int, batchSize: Int)

    abstract fun evaluate(testDataset: ImageDataset, metric: Metrics): Double
}