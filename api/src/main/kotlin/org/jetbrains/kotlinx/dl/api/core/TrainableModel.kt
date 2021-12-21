/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits
import org.jetbrains.kotlinx.dl.api.core.metric.Accuracy
import org.jetbrains.kotlinx.dl.api.core.metric.EvaluationResult
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Optimizer
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.api.core.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.io.File
import java.io.FileNotFoundException

/**
 * Base abstract class for all trainable models.
 */
public abstract class TrainableModel : TensorFlowInferenceModel() {
    /** Optimization algorithm required for compiling a model, and its learning rate. */
    protected var optimizer: Optimizer = SGD(0.2f)

    /** Loss function. */
    public var loss: LossFunction = SoftmaxCrossEntropyWithLogits()

    /** Callback. */
    protected var callback: Callback = Callback()

    /** List of metrics for evaluation phase. */
    protected var metrics: List<Metric> = listOf(Accuracy())

    /** Number of classes for classification tasks. -1 is a default value for regression tasks. */
    public var numberOfClasses: Long = -1

    /** Is true when model is compiled. */
    public var isModelCompiled: Boolean = false
        internal set

    /** Is true when model is ready for forward mode. */
    public var isBuiltForForwardMode: Boolean = false
        internal set

    /**
     * Is true when model optimizer variables are initialized.
     *
     * NOTE: This flag is important for training purposes only (in training from zero to hero or transfer learning training).
     * This flag is not checked before evaluation or prediction phases.
     */
    public var isOptimizerVariableInitialized: Boolean = false
        internal set

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
     * Configures the model for training.
     *
     * NOTE: Set up [isModelCompiled] to True.
     *
     * @param [optimizer] Optimizer instance.
     * @param [loss] Loss function.
     * @param [metrics] Metrics to evaluate during training.
     * @param [callback] Callback to be used during training, evaluation and prediction phases.
     */
    public abstract fun compile(optimizer: Optimizer, loss: LossFunction, metrics: List<Metric>, callback: Callback = Callback())

    /**
     * Trains the model for a fixed number of [epochs] (iterations over a dataset).
     *
     * @param [dataset] The train dataset that combines input data (X) and target data (Y).
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [batchSize] Number of samples per gradient update.
     * True (default) = Weights are initialized at the beginning of the training phase.
     * False = Weights are not initialized during training phase. It should be initialized before (via transfer learning or init() method call).
     *
     * @return A [TrainingHistory] object. Its [TrainingHistory.batchHistory] attribute is a record of training loss values and metrics values per each batch and epoch.
     */
    public abstract fun fit(
        dataset: Dataset,
        epochs: Int = 5,
        batchSize: Int = 32
    ): TrainingHistory

    /**
     * Trains the model for a fixed number of [epochs] (iterations over a dataset).
     *
     * @param [trainingDataset] The train dataset that combines input data (X) and target data (Y).
     * @param [validationDataset] The validation dataset that combines input data (X) and target data (Y).
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [trainBatchSize] Number of samples per gradient update.
     * @param [validationBatchSize] Number of samples per validation batch.
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
        validationBatchSize: Int = 256
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
     * @return Array of labels. The length is equal to the Number of samples on the [dataset].
     */
    public abstract fun predict(dataset: Dataset, batchSize: Int): IntArray

    /**
     * Generates output predictions for the input samples.
     * Each prediction is a vector of probabilities instead of specific class in [predict] method.
     *
     * @param [dataset] Data to predict on.
     * @param [batchSize] Number of samples per batch of computation.
     * @return Array of labels. All labels are vectors that represents the probability distributions of a list of potential outcomes. The length is equal to the Number of samples on the [dataset].
     */
    public abstract fun predictSoftly(dataset: Dataset, batchSize: Int): Array<FloatArray>

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
     * @throws [FileNotFoundException] If [modelDirectory] does not contain all required files.
     */
    public abstract fun save(
        modelDirectory: File,
        savingFormat: SavingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
        saveOptimizerState: Boolean = false,
        writingMode: WritingMode = WritingMode.FAIL_IF_EXISTS
    )

    /**
     * Loads variable data from .txt files.
     *
     * @param [modelDirectory] Path to directory with TensorFlow graph and variable data.
     * @param [loadOptimizerState] Loads optimizer internal variables data, if true.
     * @throws [FileNotFoundException] If file with weights is not found.
     */
    public open fun loadWeights(
        modelDirectory: File,
        loadOptimizerState: Boolean = false
    ) {
        loadVariablesFromTxt(modelDirectory.absolutePath, loadOptimizerState)
    }

    /**
     * Trains the model for a fixed number of [epochs] (iterations on a dataset).
     *
     * @param [dataset] The dataset that combines input data (X) and target data (Y). It will be split on train and validation sub-datasets.
     * @param [validationRate] Number between 0.0 and 1.0. The proportion of validation data from initially passed [dataset].
     * @param [epochs] Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
     * @param [trainBatchSize] Number of samples per gradient update.
     * @param [validationBatchSize] Number of samples per validation batch.
     * @return A [TrainingHistory] object. It contains records with training/validation loss values and metrics per each batch and epoch.
     */
    public fun fit(
        dataset: OnHeapDataset,
        validationRate: Double,
        epochs: Int,
        trainBatchSize: Int,
        validationBatchSize: Int
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
            validationBatchSize
        )
    }

    public override fun close() {
        super.close()
    }

    /**
     * Returns model summary.
     *
     * @return model summary
     */
    public abstract fun summary(): ModelSummary

    override fun toString(): String {
        return "TrainableModel(numberOfClasses=$numberOfClasses) ${super.toString()}"
    }
}
