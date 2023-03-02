package examples.inference.saveload

import examples.inference.lenet5
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import java.io.File


private const val MODEL_SAVE_PATH = "savedmodels/relu_lenet_saveload"

/**
 * This examples demonstrates running Save and Load for prediction on [mnist] dataset.
 */
fun reluLenetOnMnistWithIntermediateSave() {
    val (train, test) = mnist()
    SaveTrainedModelHelper().trainAndSave(
        train, test, lenet5(),
        MODEL_SAVE_PATH, 0.7
    )
    Sequential.loadDefaultModelConfiguration(File(MODEL_SAVE_PATH)).use {
        it.compile(
            optimizer = SGD(learningRate = 0.3f),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.loadWeights(File(MODEL_SAVE_PATH))
        val accuracy = it.evaluate(test).metrics[Metrics.ACCURACY] ?: 0.0
        println("Accuracy is : $accuracy")
    }
}

fun main(): Unit = reluLenetOnMnistWithIntermediateSave()
