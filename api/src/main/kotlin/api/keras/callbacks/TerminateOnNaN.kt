package api.keras.callbacks

import api.keras.history.BatchTrainingEvent
import api.keras.history.TrainingHistory

class TerminateOnNaN<T : Number> : Callback<T>() {
    override fun onTrainBatchEnd(batch: Int, logs: BatchTrainingEvent?, trainingHistory: TrainingHistory) {
        val loss = logs!!.lossValue
        if (loss.isNaN() || loss == Double.POSITIVE_INFINITY || loss == Double.NEGATIVE_INFINITY) {
            this.model.logger.info { "Batch $batch: Invalid loss $loss, terminating training" }
            this.model.stopTraining = true
        }
    }
}