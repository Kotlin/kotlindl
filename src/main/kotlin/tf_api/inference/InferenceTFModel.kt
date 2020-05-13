package tf_api.inference

import org.tensorflow.SavedModelBundle
import org.tensorflow.Session
import org.tensorflow.Tensor
import tf_api.KGraph
import tf_api.keras.Input
import tf_api.keras.Output
import tf_api.keras.metric.Metrics
import util.MnistUtils

open class InferenceTFModel() : AutoCloseable {
    protected lateinit var session: Session
    private lateinit var bundle: SavedModelBundle
    private lateinit var kGraph: KGraph
    private lateinit var reshape: (DoubleArray) -> Tensor<*>?
    private lateinit var input: Input
    private lateinit var output: Output

    fun predict(image: MnistUtils.Image): LongArray {
        return predictOnImage(image)
    }

    private fun predictOnImage(
        image: MnistUtils.Image
    ): LongArray {
        val runner = session.runner()
        return runner.feed(input.tfName, reshape(image.pixels))
            .fetch(output.tfName)
            .run()[0]
            .copyTo(LongArray(1))
    }

    fun predictAll(images: List<MnistUtils.Image>): List<Double> {
        val predictedLabels: MutableList<Double> = mutableListOf()

        for (image in images) {
            val predictedLabel = predictOnImage(image)
            predictedLabels.add(predictedLabel[0].toDouble())
        }

        return predictedLabels
    }

    fun predict(inputData: DoubleArray): Double {
        return 0.0
    }

    fun predict(inputData: List<DoubleArray>): List<Double> {
        return listOf()
    }

    fun evaluate(
        testImages: MutableList<MnistUtils.LabeledImage>,
        metric: Metrics
    ): Double {

        return if (metric == Metrics.ACCURACY) {
            var counter = 0
            for (image in testImages) {
                val result = predictOnImage(image)
                if (result[0].toInt() == image.label)
                    counter++
            }

            (counter.toDouble() / testImages.size)
        } else {
            Double.NaN
        }
    }


    fun loadModel(pathToModel: String): InferenceTFModel {
        bundle = SavedModelBundle.load(pathToModel, "serve")
        session = bundle.session()
        val graph = bundle.graph()
        val graphDef = graph.toGraphDef()
        kGraph = KGraph(graphDef)
        return this
    }

    override fun toString(): String {
        return "Model contains $kGraph"
    }

    override fun close() {
        session.close()
        bundle.close()
    }

    fun input(inputOp: Input) {
        input = inputOp

    }

    fun output(outputOp: Output) {
        output = outputOp
    }

    fun reshape(function: (DoubleArray) -> Tensor<*>?) {
        reshape = function
    }


}

fun prepareModelForInference(init: InferenceTFModel.() -> Unit): InferenceTFModel = InferenceTFModel()
    .apply(init)