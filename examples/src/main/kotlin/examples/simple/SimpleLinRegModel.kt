package examples.simple

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.Dataset
import api.keras.initializers.HeNormal
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.SGD
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 5
private const val EPOCHS = 30
private const val TRAINING_BATCH_SIZE = 5

private val model = Sequential.of(
    Input(4),
    //Dense(4, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(1, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

/**
 * Doesn't work due to incorrect loss functions for regression
 */
fun main() {
    val rnd = Random(SEED)
    val data = Array(1000) { doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0) }
    for (i in data.indices) {
        data[i][1] = 2 * (rnd.nextDouble() - 0.5)
        data[i][2] = 2 * (rnd.nextDouble() - 0.5)
        data[i][3] = 2 * (rnd.nextDouble() - 0.5)
        data[i][4] = 2 * (rnd.nextDouble() - 0.5)
        data[i][0] = data[i][1] - 2 * data[i][2] + 1.5 * data[i][3] - 0.95 * data[i][4] + rnd.nextDouble(0.1)
    }

    data.shuffle()

    fun extractX(): Array<FloatArray> {
        val init: (index: Int) -> FloatArray = { index ->
            floatArrayOf(
                data[index][1].toFloat(),
                data[index][2].toFloat(),
                data[index][3].toFloat(),
                data[index][4].toFloat()
            )
        }
        return Array(data.size, init = init)
    }

    fun extractY(): Array<FloatArray> {
        val labels = Array(data.size) { FloatArray(1) { 0.0f } }
        for (i in labels.indices) {
            labels[i][0] = data[i][0].toFloat()
        }

        return labels
    }

    val dataset = Dataset.create(
        ::extractX,
        ::extractY
    )

    val (train, test) = dataset.split(0.9)

    model.compile(
        optimizer = SGD(learningRate = 0.001f),
        loss = LossFunctions.ABSOLUTE_DIFFERENCE,
        metric = Metrics.MSE
    )

    model.summary()
    model.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = false)

    val mse = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MSE]
    println(model.getLayer("dense_1").getWeights()[0].contentDeepToString())

    model.close()

    println("MSE: $mse")
}


