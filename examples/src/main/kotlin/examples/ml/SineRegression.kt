import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 1000
private const val TRAINING_BATCH_SIZE = 100

private val model = Sequential.of(
    Input(1, name = "input_1"),
    Dense(20, Activations.Relu, kernelInitializer = HeNormal(), biasInitializer = HeUniform(), name = "dense_1"),
    Dense(20, Activations.Relu, kernelInitializer = HeNormal(), biasInitializer = HeUniform(), name = "dense_2"),
    Dense(1, Activations.Linear, name = "dense_3")
)

fun main() {
    val input = prepareInput()

    val (train, test) = input.split(0.8)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.MAE
        )

        it.summary()

        it.fit(
            dataset = train,
            epochs = 10,
            batchSize = 100
        )

        val evaluationResult = it.evaluate(dataset = test, batchSize = 100)//.metrics[Metrics.MAE]

        println("evaluationResult: $evaluationResult")

        println("Weights: " + it.getLayer("dense_3").weights["dense_3_dense_kernel"].contentDeepToString())
        println("Bias: " + it.getLayer("dense_3").weights["dense_3_dense_bias"].contentDeepToString())

        repeat(100) { id ->
            val xReal = test.getX(id)
            val yReal = test.getY(id)

            val yPred2 = it.predict(xReal) // always returns 0
            val yPred3 = it.predictSoftly(xReal) // returns value oscillating around 1.0

            println("xReal: ${xReal[0]}, yReal: $yReal, yPred2: $yPred2, yPred3: ${yPred3[0]}")
        }
    }
}

fun prepareInput(): Dataset {
    val sampleCount = 100000

    val x = Array(sampleCount) { FloatArray(1) }
    val y = FloatArray(sampleCount)

    repeat(sampleCount) {
        val xSample = Random.nextDouble(0.0, Math.PI * 2).toFloat()
        val ySample = sin(xSample)

        x[it][0] = xSample
        y[it] = ySample
    }

    return OnHeapDataset.create(x, y)
}


