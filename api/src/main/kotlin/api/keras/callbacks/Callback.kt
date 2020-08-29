package api.keras.callbacks

import api.TrainingEvent
import api.keras.Sequential

open class Callback<T : Number> {
    lateinit var model: Sequential<T>

    /**
     * Called at the start of an epoch. This method should only be called during
     * TRAIN mode. This method is empty. Extend this class to handle this event.
     *
     * @param epoch index of epoch.
     * @param logs metric results
     */
    open fun onEpochBegin(epoch: Int, logs: TrainingEvent?) {}

    /**
     * Called at the end of an epoch.This method should only be called during
     * TRAIN mode. This method is empty. Extend this class to handle this event.
     *
     * @param epoch index of epoch.
     * @param logs metric results for this training epoch, and for the
     * validation epoch if validation is performed. Validation result keys are
     * prefixed with `val_`.
     */
    open fun onEpochEnd(epoch: Int, logs: TrainingEvent?) {}

    /**
     * Called at the beginning of a training batch in `fit` methods. This method
     * is empty. Extend this class to handle this event.
     *
     * @param batch the batch index
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    open fun onTrainBatchBegin(batch: Int, logs: TrainingEvent?) {}

    /**
     * Called at the end of a training batch in `fit` methods. This method is
     * empty. Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Metric results for this batch.
     */
    open fun onTrainBatchEnd(batch: Int, logs: TrainingEvent?) {}

    /**
     * Called at the beginning of training. This method is empty. Extend this
     * class to handle this event.
     *
     * @param logs metric results
     */
    open fun onTrainBegin(logs: TrainingEvent?) {}

    /**
     * Called at the end of training. This method is empty. Extend this class to
     * handle this event.
     *
     * @param logs metric results
     */
    open fun onTrainEnd(logs: Map<String, Number>) {}

    /**
     * Called at the beginning of a batch in `evaluate` methods. Also called at
     * the beginning of a validation batch in the `fit` methods, if validation
     * data is provided. This method is empty. Extend this class to handle this
     * event.
     *
     * @param batch the batch number
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    open fun onTestBatchBegin(batch: Int, logs: Map<String, Number>) {}

    /**
     * Called at the end of a batch in `evaluate` methods. Also called at the
     * end of a validation batch in the `fit` methods, if validation data is
     * provided.
     *
     * This method is empty. Extend this class to handle this event.
     *
     * @param batch the batch number
     * @param logs Metric results for this batch.
     */
    open fun onTestBatchEnd(batch: Int, logs: Map<String, Number>) {}

    /**
     * Called at the beginning of evaluation or validation. This method is
     * empty. Extend this class to handle this event.
     *
     * @param logs metric results
     */
    open fun onTestBegin(logs: Map<String, Number>) {}

    /**
     * Called at the end of evaluation or validation. This method is empty.
     * Extend this class to handle this event.
     *
     * @param logs metric results
     */
    open fun onTestEnd(logs: Map<String, Number>) {}

    /**
     * Called at the beginning of a batch in `predict` methods. This method is
     * empty. Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Has keys `batch` and `size` representing the current batch
     * number and the size of the batch.
     */
    open fun onPredictBatchBegin(batch: Int, logs: Map<String, Number>) {}

    /**
     * Called at the end of a batch in `predict` methods. This method is empty.
     * Extend this class to handle this event.
     *
     * @param batch index of batch within the current epoch.
     * @param logs Metric results for this batch.
     */
    open fun onPredictBatchEnd(batch: Int, logs: Map<String, Number>) {}

    /**
     * Called at the beginning of prediction. This method is empty. Extend this
     * class to handle this event.
     *
     * @param logs metric results
     */
    open fun onPredictBegin(logs: Map<String, Number>) {}

    /**
     * Called at the end of prediction. This method is empty. Extend this class
     * to handle this event.
     *
     * @param logs metric results
     */
    open fun onPredictEnd(logs: Map<String, Number>) {}
}