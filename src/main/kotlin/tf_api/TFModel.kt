package tf_api

import tf_api.inference.InferenceTFModel
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.Optimizer

abstract class TFModel<T : Number> : InferenceTFModel() {
    protected var isDebugMode = false

    /**
     * Configures the model for training.
     *
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    abstract fun compile(
        optimizer: Optimizer<T>,
        loss: LossFunctions = LossFunctions.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        metric: Metrics = Metrics.ACCURACY
    )

    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     *
     * @batchSize - Number of samples per gradient update. If unspecified, batch_size will default to 32.
     * @epochs - Number of epochs to train the model. An epoch is an iteration over the entire x and y (or whole dataset) data provided.
     * @verbose - Verbosity mode. Silent, one line per batch or one line per epoch.
     */
    abstract fun fit(
        dataset: ImageDataset,
        epochs: Int = 10,
        batchSize: Int = 32,
        verbose: Boolean
    ): TrainingHistory

    abstract fun evaluate(
        dataset: ImageDataset,
        metric: Metrics,
        batchSize: Int
    ): Double

    abstract fun predict(dataset: ImageDataset, batchSize: Int): IntArray

    abstract fun predict(image: FloatArray): Int
}