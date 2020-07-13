package api

import api.inference.InferenceModel
import api.keras.EvaluationResult
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Optimizer

abstract class TrainableTFModel<T : Number> : InferenceModel() {
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


    abstract fun fit(
        trainingDataset: ImageDataset,
        validationDataset: ImageDataset,
        epochs: Int = 10,
        trainBatchSize: Int = 32,
        validationBatchSize: Int = 256,
        verbose: Boolean
    ): TrainingHistory

    /**
     * Returns the metrics and loss values for the model in test (evaluation) mode.
     *
     * @param [dataset] The train dataset that combines input data (X) and target data (Y).
     * @param [batchSize] Number of samples per batch of computation.
     *
     * @return Value of calculated metric and loss values.
     */
    abstract fun evaluate(
        dataset: ImageDataset,
        batchSize: Int = 256
    ): EvaluationResult

    abstract fun predictAll(dataset: ImageDataset, batchSize: Int): IntArray

    // TODO: up from sequential common implementations
    abstract override fun predict(image: FloatArray): Int

    /**
     * Saves the model as graph and weights.
     */
    abstract fun save(pathToModelDirectory: String)


    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }

    abstract fun predictAndGetActivations(image: FloatArray): Pair<Int, List<*>>

    abstract fun predictSoftly(image: FloatArray): FloatArray

    abstract fun predictSoftlyAndGetActivations(
        image: FloatArray,
        visualizationIsEnabled: Boolean
    ): Pair<FloatArray, List<*>>

    fun fit(
        dataset: ImageDataset,
        validationRate: Double,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        verbose: Boolean
    ): TrainingHistory {
        assert(validationRate > 0.0 && validationRate < 1.0)
        val (validation, train) = dataset.split(validationRate)

        return fit(train, validation, epochs, trainBatchSize, validationBatchSize, verbose)
    }
}