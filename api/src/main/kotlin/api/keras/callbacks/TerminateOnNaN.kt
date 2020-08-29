package api.keras.callbacks

import api.TrainingEvent

class TerminateOnNaN<T : Number> : Callback<T>() {
    override fun onTrainBatchEnd(batch: Int, logs: TrainingEvent?) {
        val loss = logs!!.lossValue
        if (loss.isNaN() || loss == Double.POSITIVE_INFINITY || loss == Double.NEGATIVE_INFINITY) {
            this.model.logger.info { "Batch $batch: Invalid loss $loss, terminating training" }
            this.model.stopTraining = true
        }
    }
}