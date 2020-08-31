package api.keras.history

import java.util.*

class History {
    private val history: MutableList<BatchEvent> = mutableListOf()

    private val historyByBatch: TreeMap<Int, BatchEvent> = TreeMap()

    fun appendBatch(batch: Int, lossValue: Double, metricValue: Double) {
        val newEvent = BatchEvent(batch, lossValue, metricValue)
        addNewEpochEvent(newEvent, batch)
    }

    fun appendBatch(batchEvent: BatchEvent) {
        addNewEpochEvent(batchEvent, batchEvent.batch)
    }


    private fun addNewEpochEvent(batchEvent: BatchEvent, batch: Int) {
        history.add(batchEvent)
        historyByBatch[batch] = batchEvent
    }

    fun lastBatchEvent(): BatchEvent {
        return historyByBatch.lastEntry().value
    }
}

class BatchEvent(val batch: Int, val lossValue: Double, val metricValue: Double)
