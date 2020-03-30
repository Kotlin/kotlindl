package tf_api

import org.tensorflow.SavedModelBundle
import org.tensorflow.Session
import org.tensorflow.Tensor
import util.MnistUtils

class TFModel() : AutoCloseable {
    private lateinit var session: Session
    private lateinit var bundle: SavedModelBundle
    private lateinit var kGraph: KGraph
    private lateinit var reshape: (DoubleArray) -> Tensor<*>?

    fun predict(image: MnistUtils.Image): LongArray {
        return predictOnImage(image)
    }

    private fun predictOnImage(
        image: MnistUtils.Image
    ): LongArray {
        val runner = session.runner()
        return runner.feed("Placeholder", reshape(image.pixels))
            .fetch("ArgMax")
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

    /* companion object {
         fun loadGraph(pathToModel: String): TFModel {
             val graphDef = File(pathToModel).readBytes()

             return TFModel(KGraph(graphDef))
         }


     }*/

    public fun evaluateTFModel(
        images: MutableList<MnistUtils.LabeledImage>,
        metric: Metrics
    ): Double {

        return if (metric == Metrics.ACCURACY) {
            var counter = 0
            for (image in images) {
                val result = predictOnImage(image)
                if (result[0].toInt() == image.label)
                    counter++
            }

            (counter.toDouble() / images.size)
        } else {
            Double.NaN
        }


    }


    fun loadModel(pathToModel: String): TFModel {
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

    fun input(placeholder: Input) {

    }

    fun output(argmax: Output) {

    }

    fun reshape(function: (DoubleArray) -> Tensor<*>?) {
        reshape = function
    }
}