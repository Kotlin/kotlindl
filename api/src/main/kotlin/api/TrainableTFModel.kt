package api

import api.inference.savedmodel.InferenceModel
import api.keras.EvaluationResult
import api.keras.ModelFormat
import api.keras.callbacks.Callback
import api.keras.dataset.Dataset
import api.keras.history.TrainingHistory
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Optimizer
import api.keras.optimizers.SGD
import org.tensorflow.Operand

abstract class TrainableTFModel : InferenceModel() {
    protected var isDebugMode = false

    /** Optimizer. Approach how aggressively to update the weights. */
    protected var optimizer: Optimizer = SGD(0.2f)

    /** Loss function. */
    protected var loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS

    /** Callback. */
    protected var callback: Callback = Callback()

    /** Metric on validation dataset for training phase. */
    protected var metric: Metrics = Metrics.ACCURACY

    /** List of metrics for evaluation phase. */
    protected var metrics: List<Metrics> = listOf(Metrics.ACCURACY)

    /** TensorFlow operand for prediction phase. */
    protected lateinit var yPred: Operand<Float>

    /** TensorFlow operand for X data. */
    protected lateinit var xOp: Operand<Float>

    /** TensorFlow operand for Y data. */
    protected lateinit var yOp: Operand<Float>

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
        optimizer: Optimizer,
        loss: LossFunctions = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric: Metrics = Metrics.ACCURACY,
        callback: Callback = Callback()
    )

    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     *
     * @batchSize - Number of samples per gradient update. If unspecified, batch_size will default to 32.
     * @epochs - Number of epochs to train the model. An epoch is an iteration over the entire x and y (or whole dataset) data provided.
     * @verbose - Verbosity mode. Silent, one line per batch or one line per epoch.
     */
    abstract fun fit(
        dataset: Dataset,
        epochs: Int = 10,
        batchSize: Int = 32,
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
    ): TrainingHistory

    abstract fun fit(
        trainingDataset: Dataset,
        validationDataset: Dataset,
        epochs: Int = 10,
        trainBatchSize: Int = 32,
        validationBatchSize: Int = 256,
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
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
        dataset: Dataset,
        batchSize: Int = 256
    ): EvaluationResult

    abstract fun predictAll(dataset: Dataset, batchSize: Int): IntArray

    abstract override fun predict(image: FloatArray): Int

    abstract fun predict(image: FloatArray, predictionTensorName: String): Int

    /**
     * Saves the model as graph and weights.
     */
    abstract fun save(
        pathToModelDirectory: String,
        modelFormat: ModelFormat = ModelFormat.TF_GRAPH_CUSTOM_VARIABLES,
        saveOptimizerState: Boolean = false,
        modelWritingMode: ModelWritingMode = ModelWritingMode.FAIL_IF_EXISTS
    )

    override fun loadVariablesFromTxtFiles(pathToModelDirectory: String, loadOptimizerState: Boolean) {

    }

    fun getDType(): Class<Float> {
        return Float::class.javaObjectType
    }

    abstract fun predictAndGetActivations(
        image: FloatArray,
        predictionTensorName: String = OUTPUT_NAME
    ): Pair<Int, List<*>>

    abstract fun predictSoftly(image: FloatArray, predictionTensorName: String = OUTPUT_NAME): FloatArray

    abstract fun predictSoftlyAndGetActivations(
        image: FloatArray,
        visualizationIsEnabled: Boolean,
        predictionTensorName: String = OUTPUT_NAME
    ): Pair<FloatArray, List<*>>

    fun fit(
        dataset: Dataset,
        validationRate: Double,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
    ): TrainingHistory {
        require(validationRate > 0.0 && validationRate < 1.0) { "Validation rate should be more than 0.0 and less than 1.0. The passed rare is: ${validationRate}" }
        val (validation, train) = dataset.split(validationRate)

        return fit(
            train,
            validation,
            epochs,
            trainBatchSize,
            validationBatchSize,
            verbose,
            isWeightsInitRequired,
            isOptimizerInitRequired
        )
    }
}