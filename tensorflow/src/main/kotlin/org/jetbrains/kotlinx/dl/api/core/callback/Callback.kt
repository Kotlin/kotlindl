/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.callback

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.history.*

/**
 * Base class used to build new callbacks.
 *
 * NOTE: This class contains empty methods, inherit it and override if you need functionality.
 *
 * Callback methods are called during training, evaluation, prediction phases on each batch, epoch, start and end of the specific phase.
 */
public open class Callback {
    /** Sequential model, accessible inside callback methods. */
    internal lateinit var model: GraphTrainableModel

    /**
     * Called at the start of an epoch during training phase.
     *
     * @param [epoch] index of epoch.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onEpochBegin(epoch: Int, logs: TrainingHistory) {}

    /**
     * Called at the end of an epoch during training phase.
     *
     * @param [epoch] index of epoch.
     * @param [event] metric results for this training epoch, and for the
     * validation epoch if validation is performed.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {}

    /**
     * Called at the beginning of a batch during training phase.
     *
     * @param [batch] the batch index.
     * @param [batchSize] Number of samples in the current batch.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onTrainBatchBegin(batch: Int, batchSize: Int, logs: TrainingHistory) {}

    /**
     * Called at the end of a batch during training phase.
     *
     * @param [batch] index of batch within the current epoch.
     * @param [batchSize] Number of samples in the current batch.
     * @param [event] Metric and loss values for this batch.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onTrainBatchEnd(batch: Int, batchSize: Int, event: BatchTrainingEvent, logs: TrainingHistory) {}

    /**
     * Called at the beginning of training.
     */
    public open fun onTrainBegin() {}

    /**
     * Called at the end of training. This method is empty. Extend this class to
     * handle this event.
     *
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onTrainEnd(logs: TrainingHistory) {}

    /**
     * Called at the beginning of a batch during evaluation phase. Also called at
     * the beginning of a validation batch during validation phase, if validation
     * data is provided.
     *
     * @param [batch] the batch number
     * @param [batchSize] Number of samples in the current batch.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onTestBatchBegin(batch: Int, batchSize: Int, logs: History) {}

    /**
     * Called at the end of a batch during evaluation phase. Also called at the
     * end of a validation batch during validation phase, if validation data is
     * provided.
     *
     * @param [batch] the batch number
     * @param [batchSize] Number of samples in the current batch.
     * @param [event] Metric and loss values for this batch.
     * @param [logs] training history, containing full information about previous epochs.
     */
    public open fun onTestBatchEnd(batch: Int, batchSize: Int, event: BatchEvent?, logs: History) {}

    /**
     * Called at the beginning of evaluation or validation.
     */
    public open fun onTestBegin() {}

    /**
     * Called at the end of evaluation or validation.
     *
     * @param [logs] evaluation history, containing full information about previous batches.
     */
    public open fun onTestEnd(logs: History) {}

    /**
     * Called at the beginning of a batch during prediction phase.
     *
     * @param [batch] index of batch.
     * @param [batchSize] Number of samples in the current batch.
     */
    public open fun onPredictBatchBegin(batch: Int, batchSize: Int) {}

    /**
     * Called at the end of a batch during prediction phase.
     *
     * @param [batch] index of batch within the current epoch.
     * @param [batchSize] Number of samples in the current batch.
     */
    public open fun onPredictBatchEnd(batch: Int, batchSize: Int) {}

    /**
     * Called at the beginning of prediction.
     */
    public open fun onPredictBegin() {}

    /**
     * Called at the end of prediction.
     */
    public open fun onPredictEnd() {}
}
