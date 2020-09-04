package api.keras.history

import java.util.*
import kotlin.reflect.KProperty1

class TrainingHistory {
    private val _batchHistory: MutableList<BatchTrainingEvent> = mutableListOf()

    val batchHistory: List<BatchTrainingEvent>
        get() = Collections.unmodifiableList(_batchHistory)

    private val historyByEpochAndBatch: TreeMap<Int, TreeMap<Int, BatchTrainingEvent>?> = TreeMap()

    private val _epochHistory: MutableList<EpochTrainingEvent> = mutableListOf()

    private val _historyByEpoch: TreeMap<Int, EpochTrainingEvent> = TreeMap()

    val epochHistory: List<EpochTrainingEvent>
        get() = Collections.unmodifiableList(_epochHistory)

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

    fun lastBatchEvent(): BatchTrainingEvent {
        return historyByEpochAndBatch.lastEntry().value!!.lastEntry().value
    }

    fun lastEpochEvent(): EpochTrainingEvent {
        return _historyByEpoch.lastEntry().value
    }

    fun eventsByEpoch(epoch: Int): TreeMap<Int, BatchTrainingEvent>? {
        return historyByEpochAndBatch[epoch]
    }

    operator fun get(desiredField: KProperty1<EpochTrainingEvent, Double>): DoubleArray {
        val result = DoubleArray(_epochHistory.size)
        for (i in 0 until _epochHistory.size) {
            result[i] = desiredField.invoke(_epochHistory[i]!!)
        }
        return result
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