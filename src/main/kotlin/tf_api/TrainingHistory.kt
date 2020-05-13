package tf_api

class TrainingHistory {
    var history: List<TrainingEvent> = emptyList()

    fun append(epoch: Int, batch: Int, lossValue: Double, metricValue: Double) {
        history = history + TrainingEvent(epoch, batch, lossValue, metricValue)
    }
}

class TrainingEvent(val epoch: Int, val batch: Int, val lossValue: Double, val metricValue: Double)
