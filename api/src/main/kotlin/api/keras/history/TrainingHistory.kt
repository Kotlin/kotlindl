package api.keras.history

import java.util.*

class TrainingHistory {
    private val _history: MutableList<BatchTrainingEvent> = mutableListOf()

    val history: List<BatchTrainingEvent>
        get() = Collections.unmodifiableList(_history)

    private val historyByEpochAndBatch: TreeMap<Int, TreeMap<Int, BatchTrainingEvent>?> = TreeMap()

    private val historyInEpochs: MutableList<EpochTrainingEvent> = mutableListOf()

    private val _historyByEpoch: TreeMap<Int, EpochTrainingEvent> = TreeMap()

    val historyByEpoch: Map<Int, EpochTrainingEvent>
        get() = Collections.unmodifiableMap(_historyByEpoch)

    fun appendBatch(epoch: Int, batch: Int, lossValue: Double, metricValue: Double) {
        val newEvent = BatchTrainingEvent(epoch, batch, lossValue, metricValue)
        addNewBatchEvent(newEvent, epoch, batch)
    }

    fun appendBatch(batchTrainingEvent: BatchTrainingEvent) {
        addNewBatchEvent(batchTrainingEvent, batchTrainingEvent.epoch, batchTrainingEvent.batch)
    }

    fun appendEpoch(
        epoch: Int,
        lossValue: Double,
        metricValue: Double,
        valLossValue: Double?,
        valMetricValue: Double?
    ) {
        val newEvent = EpochTrainingEvent(epoch, lossValue, metricValue, valLossValue, valMetricValue)
        addNewEpochEvent(newEvent, epoch)
    }

    fun appendEpoch(epochTrainingEvent: EpochTrainingEvent) {
        addNewEpochEvent(epochTrainingEvent, epochTrainingEvent.epoch)
    }

    private fun addNewBatchEvent(newEventBatch: BatchTrainingEvent, epoch: Int, batch: Int) {
        _history.add(newEventBatch)
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
        historyInEpochs.add(newEventEpoch)
        _historyByEpoch[epoch] = newEventEpoch
    }

    fun lastBatchEvent(): BatchTrainingEvent {
        return historyByEpochAndBatch.lastEntry().value!!.lastEntry().value
    }

    fun lastEpochEvent(): EpochTrainingEvent {
        return _historyByEpoch.lastEntry().value
    }

    fun eventsByEpoch(epoch: Int): TreeMap<Int, BatchTrainingEvent>? {
        return historyByEpochAndBatch[epoch]
    }
}

class BatchTrainingEvent(val epoch: Int, val batch: Int, val lossValue: Double, val metricValue: Double) {
    override fun toString(): String {
        return "BatchTrainingEvent(epoch=$epoch, batch=$batch, lossValue=$lossValue, metricValue=$metricValue)"
    }
}

class EpochTrainingEvent(
    val epoch: Int,
    val lossValue: Double,
    val metricValue: Double,
    var valLossValue: Double?,
    var valMetricValue: Double?

) {
    override fun toString(): String {
        return "EpochTrainingEvent(epoch=$epoch, lossValue=$lossValue, metricValue=$metricValue, valLossValue=$valLossValue, valMetricValue=$valMetricValue)"
    }
}