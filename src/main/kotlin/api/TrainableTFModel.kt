package api

import api.inference.InferenceModel
import api.keras.EvaluationResult
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Optimizer
import api.keras.optimizers.SGD
import org.tensorflow.Operand

abstract class TrainableTFModel<T : Number> : InferenceModel<T>() {
    protected var isDebugMode = false

    /** Optimizer. Approach how aggressively to update the weights. */
    protected var optimizer: Optimizer<T> = SGD(0.2f)

    /** Loss function. */
    protected var loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS

    /** Metric on validation dataset for training phase. */
    protected var metric: Metrics = Metrics.ACCURACY

    /** List of metrics for evaluation phase. */
    protected var metrics: List<Metrics> = listOf(Metrics.ACCURACY)

    /** TensorFlow operand for prediction phase. */
    protected lateinit var yPred: Operand<T>

    /** TensorFlow operand for X data. */
    protected lateinit var xOp: Operand<T>

    /** TensorFlow operand for Y data. */
    protected lateinit var yOp: Operand<T>

    /** Amount of classes for classification tasks. -1 is a default value for regression tasks. */
    protected var amountOfClasses: Long = -1

    /**
     * Configures the model for training.
     *
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    abstract fun compile(
        optimizer: Optimizer<T>,
        loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
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
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true
    ): TrainingHistory

    abstract fun fit(
        trainingDataset: ImageDataset,
        validationDataset: ImageDataset,
        epochs: Int = 10,
        trainBatchSize: Int = 32,
        validationBatchSize: Int = 256,
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true
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
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true
    ): TrainingHistory {
        require(validationRate > 0.0 && validationRate < 1.0) { "Validation rate should be more than 0.0 and less than 1.0. The passed rare is: ${validationRate}" }
        val (validation, train) = dataset.split(validationRate)

        return fit(train, validation, epochs, trainBatchSize, validationBatchSize, verbose, isWeightsInitRequired)
    }
}