package api.keras.callbacks

import api.keras.history.BatchTrainingEvent
import api.keras.history.TrainingHistory

class TerminateOnNaN : Callback() {
    override fun onTrainBatchEnd(batch: Int, event: BatchTrainingEvent?, logs: TrainingHistory) {
        val loss = event!!.lossValue
        if (loss.isNaN() || loss == Double.POSITIVE_INFINITY || loss == Double.NEGATIVE_INFINITY) {
            this.model.logger.info { "Batch $batch: Invalid loss $loss, terminating training" }
            this.model.stopTraining = true
        }
    }
}