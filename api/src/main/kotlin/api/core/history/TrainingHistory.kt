package api.core.history

import java.util.*
import kotlin.reflect.KProperty1

/**
 * Contains all recorded batch events as a list of [BatchTrainingEvent] objects and epoch events as a list of [EpochTrainingEvent] objects.
 *
 * NOTE: Used to record [BatchTrainingEvent] and [EpochTrainingEvent] during training phase.
 */
class TrainingHistory {
    /** Hidden batch history. */
    private val _batchHistory: MutableList<BatchTrainingEvent> = mutableListOf()

    /**
     * Batch history.
     *
     * @return Unmodifiable list of [BatchTrainingEvent].
     */
    val batchHistory: List<BatchTrainingEvent>
        get() = Collections.unmodifiableList(_batchHistory)

    /** Batch history indexed by batch and epoch indices. */
    private val historyByEpochAndBatch: TreeMap<Int, TreeMap<Int, BatchTrainingEvent>?> = TreeMap()

    /** Hidden epoch history. */
    private val _epochHistory: MutableList<EpochTrainingEvent> = mutableListOf()

    /** Epoch history indexed by epoch indices. */
    private val _historyByEpoch: TreeMap<Int, EpochTrainingEvent> = TreeMap()

    /**
     * Epoch history.
     *
     * @return Unmodifiable list of [EpochTrainingEvent].
     */
    val epochHistory: List<EpochTrainingEvent>
        get() = Collections.unmodifiableList(_epochHistory)

    /**
     * Appends tracked data from one batch event.
     *
     * @param epochIndex Epoch index.
     * @param batchIndex Epoch index.
     * @param lossValue Value of loss function on training dataset.
     * @param metricValue Value of metric function on training dataset.
     */
    fun appendBatch(epochIndex: Int, batchIndex: Int, lossValue: Double, metricValue: Double) {
        val newEvent = BatchTrainingEvent(epochIndex, batchIndex, lossValue, metricValue)
        addNewBatchEvent(newEvent, epochIndex, batchIndex)
    }

    /**
     * Appends one [BatchTrainingEvent].
     */
    fun appendBatch(batchTrainingEvent: BatchTrainingEvent) {
        addNewBatchEvent(batchTrainingEvent, batchTrainingEvent.epochIndex, batchTrainingEvent.batchIndex)
    }

    /**
     * Appends tracked data from one epoch event.
     *
     * @param epochIndex Epoch index.
     * @param lossValue Value of loss function on training dataset.
     * @param metricValue Value of metric function on training dataset.
     * @param valLossValue Value of loss function on validation dataset. Could be null, if validation phase is missed.
     * @param valMetricValue Value of metric function on validation dataset. Could be null, if validation phase is missed.
     */
    fun appendEpoch(
        epochIndex: Int,
        lossValue: Double,
        metricValue: Double,
        valLossValue: Double?,
        valMetricValue: Double?
    ) {
        val newEvent = EpochTrainingEvent(epochIndex, lossValue, metricValue, valLossValue, valMetricValue)
        addNewEpochEvent(newEvent, epochIndex)
    }

    /**
     * Appends one [EpochTrainingEvent].
     */
    fun appendEpoch(epochTrainingEvent: EpochTrainingEvent) {
        addNewEpochEvent(epochTrainingEvent, epochTrainingEvent.epochIndex)
    }

    private fun addNewBatchEvent(newEventBatch: BatchTrainingEvent, epoch: Int, batch: Int) {
        _batchHistory.add(newEventBatch)
        if (historyByEpochAndBatch.containsKey(epoch)) {
            val historyByEpoch = historyByEpochAndBatch[epoch]
            historyByEpoch!![batch] = newEventBatch
        } else {
            val historyByEpoch = TreeMap<Int, BatchTrainingEvent>()
            historyByEpoch[batch] = newEventBatch
            historyByEpochAndBatch[epoch] = historyByEpoch
        }
    }

    private fun addNewEpochEvent(newEventEpoch: EpochTrainingEvent, epoch: Int) {
        _epochHistory.add(newEventEpoch)
        _historyByEpoch[epoch] = newEventEpoch
    }

    /**
     * Returns last [BatchTrainingEvent]
     */
    fun lastBatchEvent(): BatchTrainingEvent {
        return historyByEpochAndBatch.lastEntry().value!!.lastEntry().value
    }

    /**
     * Returns last [EpochTrainingEvent].
     */
    fun lastEpochEvent(): EpochTrainingEvent {
        return _historyByEpoch.lastEntry().value
    }

    /**
     * Returns all [BatchTrainingEvent] of the specific epoch.
     *
     * @param [epochIndex] Epoch index of the required epoch to return its batch events.
     * @return Indexed and sorted [TreeMap] of [BatchTrainingEvent].
     */
    fun eventsByEpoch(epochIndex: Int): TreeMap<Int, BatchTrainingEvent>? {
        return historyByEpochAndBatch[epochIndex]
    }

    /**
     * Returns all values of one filed in [EpochTrainingEvent] for all epochs. All [EpochTrainingEvent.metricValue] for example.
     */
    operator fun get(desiredField: KProperty1<EpochTrainingEvent, Double>): DoubleArray {
        val result = DoubleArray(_epochHistory.size)
        for (i in 0 until _epochHistory.size) {
            result[i] = desiredField.invoke(_epochHistory[i])
        }
        return result
    }
}

/**
 * One record in [TrainingHistory] objects containing tracked data from one batch in the specific epoch.
 *
 * @constructor Creates [BatchTrainingEvent] from [epochIndex], [batchIndex], [lossValue], [metricValue].
 * @param epochIndex Batch index.
 * @param batchIndex Batch index.
 * @param lossValue Final value of loss function.
 * @param metricValue Final value of chosen metric.
 */
class BatchTrainingEvent(val epochIndex: Int, val batchIndex: Int, val lossValue: Double, val metricValue: Double) {
    override fun toString(): String {
        return "BatchTrainingEvent(epoch=$epochIndex, batch=$batchIndex, lossValue=$lossValue, metricValue=$metricValue)"
    }
}

/**
 * One record in [TrainingHistory] objects containing tracked data from one epoch.
 *
 * @constructor Creates [EpochTrainingEvent] from [epochIndex], [lossValue], [metricValue], [valLossValue], [valMetricValue].
 * @param epochIndex Batch index.
 * @param lossValue Value of loss function on training dataset.
 * @param metricValue Value of metric function on training dataset.
 * @param valLossValue Value of loss function on validation dataset. Could be null, if validation phase is missed.
 * @param valMetricValue Value of metric function on validation dataset. Could be null, if validation phase is missed.
 */
class EpochTrainingEvent(
    val epochIndex: Int,
    val lossValue: Double,
    val metricValue: Double,
    var valLossValue: Double?,
    var valMetricValue: Double?

) {
    override fun toString(): String {
        return "EpochTrainingEvent(epoch=$epochIndex, lossValue=$lossValue, metricValue=$metricValue, valLossValue=$valLossValue, valMetricValue=$valMetricValue)"
    }
}