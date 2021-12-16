package examples.inference.saveload

import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.BatchTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.Dataset
import java.io.File

/**
 * The object wraps the logic on a given model training up to particular accuracy on a test dataset
 * and then persist it in a file.
 */
class SaveTrainedModelHelper(private val trainBatchSize: Int = 500, private val testBatchSize: Int = 1000) {

    /**
     * Train [model] on [train] dataset and evaluate accuracy on [test] dataset until [accuracyThreshold] is reached
     * then saves model to the folder [path].
     */
    fun trainAndSave(train: Dataset, test: Dataset, model: Sequential, path: String, accuracyThreshold: Double = 0.7) {
        model.use {
            it.name = "lenet-accuracy$accuracyThreshold"
            it.compile(
                optimizer = SGD(learningRate = 0.3f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY,
                callback = object : Callback() {
                    override fun onTrainBatchEnd(
                        batch: Int,
                        batchSize: Int,
                        event: BatchTrainingEvent,
                        logs: TrainingHistory
                    ) {
                        if (event.metricValues[0] > accuracyThreshold) { // TODO: handle case with multiple metrics and accuracy metric
                            println("Stopping training at ${event.metricValues[0]} accuracy") // TODO: handle case with multiple metrics and accuracy metric
                            model.stopTraining = true
                        }
                    }
                }
            )
            it.init()
            var accuracy = 0.0
            while (accuracy < accuracyThreshold) {
                it.fit(dataset = train, epochs = 1, batchSize = trainBatchSize)
                accuracy = it.evaluate(dataset = test, batchSize = testBatchSize).metrics[Metrics.ACCURACY] ?: 0.0
                println("Accuracy: $accuracy")
            }
            model.save(
                modelDirectory = File(path),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WritingMode.OVERRIDE
            )
        }
    }
}
