package api.keras.history

import java.util.*

/**
 * Contains all recorded batch events as a list of [BatchEvent] objects.
 *
 * NOTE: Used to record [BatchEvent] during prediction and evaluation phases.
 */
class History {
    /** Plain list of batch events. */
    private val history: MutableList<BatchEvent> = mutableListOf()

    /** Indexed and sorted storage of batch events. */
    private val historyByBatch: TreeMap<Int, BatchEvent> = TreeMap()

    /**
     * Appends tracked data from one batch event.
     */
    fun appendBatch(batch: Int, lossValue: Double, metricValue: Double) {
        val newEvent = BatchEvent(batch, lossValue, metricValue)
        addNewBatchEvent(newEvent, batch)
    }

    /**
     * Appends one [BatchEvent].
     */
    fun appendBatch(batchEvent: BatchEvent) {
        addNewBatchEvent(batchEvent, batchEvent.batchIndex)
    }

    private fun addNewBatchEvent(batchEvent: BatchEvent, batchIndex: Int) {
        history.add(batchEvent)
        historyByBatch[batchIndex] = batchEvent
    }

    /**
     * Returns last [BatchEvent]
     */
    fun lastBatchEvent(): BatchEvent {
        return historyByBatch.lastEntry().value
    }
}

/**
 * One record in [History] objects containing tracked data from one batch.
 *
 * @constructor Creates [BatchEvent] from [batchIndex], [lossValue], [metricValue].
 * @param batchIndex Batch index.
 * @param lossValue Final value of loss function.
 * @param metricValue Final value of chosen metric.
 */
class BatchEvent(val batchIndex: Int, val lossValue: Double, val metricValue: Double) {
    override fun toString(): String {
        return "BatchEvent(batchIndex=$batchIndex, lossValue=$lossValue, metricValue=$metricValue)"
    }
}
