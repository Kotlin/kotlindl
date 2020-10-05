package api.core

import api.core.callback.Callback
import api.core.history.TrainingHistory
import api.core.loss.LossFunction
import api.core.loss.Losses
import api.core.loss.SoftmaxCrossEntropyWithLogits
import api.core.metric.Accuracy
import api.core.metric.EvaluationResult
import api.core.metric.Metric
import api.core.metric.Metrics
import api.core.optimizer.Optimizer
import api.core.optimizer.SGD
import api.inference.InferenceModel
import datasets.Dataset
import org.tensorflow.Operand
import java.io.File

/**
 * Base abstract class for all trainable models.
 */
public abstract class TrainableModel : InferenceModel() {
    /** Controls level of verbosity. */
    protected var isDebugMode: Boolean = false

    /** Optimization algorithm required for compiling a model, and its learning rate. */
    protected var optimizer: Optimizer = SGD(0.2f)

    /** Loss function. */
    protected var loss: LossFunction = SoftmaxCrossEntropyWithLogits()

    /** Callback. */
    protected var callback: Callback = Callback()

    /** Metric on validation dataset for training phase. */
    protected var metric: Metric = Accuracy()

    /** List of metrics for evaluation phase. */
    protected var metrics: List<Metric> = listOf(Accuracy())

    /** TensorFlow operand for prediction phase. */
    protected lateinit var yPred: Operand<Float>

    /** TensorFlow operand for X data. */
    protected lateinit var xOp: Operand<Float>

    /** TensorFlow operand for Y data. */
    protected lateinit var yOp: Operand<Float>

    /** Amount of classes for classification tasks. -1 is a default value for regression tasks. */
    protected var amountOfClasses: Long = -1

    /** Is true when model is compiled. */
    public var isModelCompiled: Boolean = false
        protected set

    /** Is true when model is initialized. */
    public var isModelInitialized: Boolean = false
        protected set

    /** Special flag for callbacks. */
    public var stopTraining: Boolean = false

    /**
     * Configures the model for training.
     *
     * NOTE: Set up [isModelCompiled] to True.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metric] Metric to evaluate during training.
     * @param [callback] Callback to be used during training, evaluation and prediction phases.
     */
    public abstract fun compile(
        optimizer: Optimizer,
        loss: Losses,
        metric: Metrics,
        callback: Callback = Callback()
    )

    /**
     * Configures the model for training.
     *
     * NOTE: Set up [isModelCompiled] to True.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metric] Metric to evaluate during training.
     * @param [callback] Callback to be used during training, evaluation and prediction phases.
     */
    public abstract fun compile(
        optimizer: Optimizer,
        loss: LossFunction,
        metric: Metric,
        callback: Callback = Callback()
    )

    /**
     * Configures the model for training.
     *
     * NOTE: Set up [isModelCompiled] to True.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metric] Metric to evaluate during training.
     * @param [callback] Callback to be used during training, evaluation and prediction phases.
     */
    public abstract fun compile(
        optimizer: Optimizer,
        loss: Losses,
        metric: Metric,
        callback: Callback = Callback()
    )

    /**
     * Configures the model for training.
     *
     * NOTE: Set up [isModelCompiled] to True.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metric] Metric to evaluate during training.
     * @param [callback] Callback to be used during training, evaluation and prediction phases.
     */
    public abstract fun compile(
        optimizer: Optimizer,
        loss: LossFunction,
        metric: Metrics,
        callback: Callback = Callback()
    )

    /**
     * Trains the model for a fixed number of [epochs] (iterations over a dataset).
     *
     * @param [dataset] The train dataset that combines input data (X) and target data (Y).
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [batchSize] Number of samples per gradient update.
     * @param [verbose] Verbosity mode. False = silent, True = one line per batch and epoch.
     * @param [isWeightsInitRequired] Weights initialization mode.
     * True (default) = Weights are initialized at the beginning of the training phase.
     * False = Weights are not initialized during training phase. It should be initialized before (via transfer learning or init() method call).
     * @param [isOptimizerInitRequired] Optimizer variables initialization mode.
     * True (default) = optimizer variables are initialized at the beginning of the training phase.
     * False = optimizer variables are not initialized during training phase. It should be initialized before (via transfer learning).
     *
     * @return A [TrainingHistory] object. Its History.history attribute is a record of training loss values and metrics values per each batch and epoch.
     */
    public abstract fun fit(
        dataset: Dataset,
        epochs: Int = 5,
        batchSize: Int = 32,
        verbose: Boolean = true,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
    ): TrainingHistory

    /**
     * Trains the model for a fixed number of [epochs] (iterations over a dataset).
     *
     * @param [trainingDataset] The train dataset that combines input data (X) and target data (Y).
     * @param [validationDataset] The validation dataset that combines input data (X) and target data (Y).
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [trainBatchSize] Number of samples per gradient update.
     * @param [validationBatchSize] Number of samples per validation batch.
     * @param [verbose] Verbosity mode. False = silent, True = one line per batch and epoch.
     * @param [isWeightsInitRequired] Weights initialization mode.
     * True (default) = Weights are initialized at the beginning of the training phase.
     * False = Weights are not initialized during training phase. It should be initialized before (via transfer learning or init() method call).
     * @param [isOptimizerInitRequired] Optimizer variables initialization mode.
     * True (default) = optimizer variables are initialized at the beginning of the training phase.
     * False = optimizer variables are not initialized during training phase. It should be initialized before (via transfer learning).
     *
     * @return A [TrainingHistory] object. It contains records with training/validation loss values and metrics per each batch and epoch.
     */
    public abstract fun fit(
        trainingDataset: Dataset,
        validationDataset: Dataset,
        epochs: Int = 5,
        trainBatchSize: Int = 32,
        validationBatchSize: Int = 256,
        verbose: Boolean = true,
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
    public abstract fun evaluate(
        dataset: Dataset,
        batchSize: Int = 256
    ): EvaluationResult

    /**
     * Generates output predictions for the input samples.
     *
     * @param [dataset] Data to predict on.
     * @param [batchSize] Number of samples per batch of computation.
     * @return Array of labels. The length is equal to the Number of sampless on the [dataset].
     */
    public abstract fun predictAll(dataset: Dataset, batchSize: Int): IntArray

    /**
     * Generates output prediction for the input sample.
     *
     * @param [inputData] Unlabeled input data to define label.
     */
    public abstract override fun predict(inputData: FloatArray): Int

    /**
     * Generates output prediction for the input sample using output of the [predictionTensorName] tensor.
     *
     * @param [inputData] Unlabeled input data to define label.
     * @param [predictionTensorName] Name of output tensor to make prediction.
     */
    public abstract fun predict(inputData: FloatArray, predictionTensorName: String): Int

    /**
     * Predicts and returns not only prediction but list of activations values from intermediate model layers
     * (for visualisation or debugging purposes).
     *
     * @param [inputData] Unlabeled input data to define label.
     * @param [predictionTensorName] Name of output tensor to make prediction.
     * @return Label (class index) and list of activations from intermediate model layers.
     */
    public abstract fun predictAndGetActivations(
        inputData: FloatArray,
        predictionTensorName: String = ""
    ): Pair<Int, List<*>>

    /**
     * Predicts and returns not only prediction but list of activations values from intermediate model layers
     * (for visualisation or debugging purposes).
     *
     * @param [inputData] Unlabeled input data to define label.
     * @param [predictionTensorName] Name of output tensor to make prediction.
     * @return Label (class index) and list of activations from intermediate model layers.
     */
    protected abstract fun predictSoftlyAndGetActivations(
        inputData: FloatArray,
        predictionTensorName: String
    ): Pair<FloatArray, List<*>>

    /**
     * Saves the model as graph and weights.
     *
     * @param [modelDirectory] Path to model directory.
     * @param [savingFormat] One of approaches to store model configurations and weights.
     * @param [saveOptimizerState] Saves internal optimizer states (variables) if true.
     * @param [writingMode] Default behaviour of handling different edge cases with existing directory before model saving.
     */
    public abstract fun save(
        modelDirectory: File,
        savingFormat: SavingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
        saveOptimizerState: Boolean = false,
        writingMode: WrintingMode = WrintingMode.FAIL_IF_EXISTS
    )

    override fun loadWeights(modelDirectory: File, loadOptimizerState: Boolean) {

    }

    /**
     * Trains the model for a fixed number of [epochs] (iterations on a dataset).
     *
     * @param [dataset] The dataset that combines input data (X) and target data (Y). It will be split on train and validation sub-datasets.
     * @param [validationRate] Number between 0.0 and 1.0. The proportion of validation data from initially passed [dataset].
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [trainBatchSize] Number of samples per gradient update.
     * @param [validationBatchSize] Number of samples per validation batch.
     * @param [verbose] Verbosity mode. False = silent, True = one line per batch and epoch.
     * @param [isWeightsInitRequired] Layer variables initialization mode.
     * True (default) = layer variables are initialized at the beginning of the training phase.
     * False = layer variables are not initialized during training phase. It should be initialized before (via transfer learning or [init] method call).
     * @param [isOptimizerInitRequired] Optimizer variables initialization mode.
     * True (default) = optimizer variables are initialized at the beginning of the training phase.
     * False = optimizer variables are not initialized during training phase. It should be initialized before (via transfer learning).
     *
     * @return A [TrainingHistory] object. It contains records with training/validation loss values and metrics per each batch and epoch.
     */
    public fun fit(
        dataset: Dataset,
        validationRate: Double,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int,
        verbose: Boolean,
        isWeightsInitRequired: Boolean = true,
        isOptimizerInitRequired: Boolean = true
    ): TrainingHistory {
        require(validationRate > 0.0 && validationRate < 1.0) {
            "Validation rate should be more than 0.0 and less than 1.0. " +
                    "The passed rare is: $validationRate"
        }
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