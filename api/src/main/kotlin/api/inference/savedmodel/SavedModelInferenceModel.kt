package api.inference.savedmodel

import api.core.KGraph
import api.inference.InferenceModel
import api.keras.metric.Metrics
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import util.MnistUtils

open class SavedModelInferenceModel : InferenceModel() {
    private lateinit var bundle: SavedModelBundle // extract for inference with SavedModelBundle and move this property to this
    private lateinit var reshape2: (DoubleArray) -> Tensor<*>?

    fun predict(image: MnistUtils.Image): LongArray {
        return predictOnImage(image)
    }

    private fun predictOnImage(
        image: MnistUtils.Image
    ): LongArray {
        val runner = session.runner()
        return runner.feed(input.tfName, reshape2(image.pixels))
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

    fun loadModel(pathToModel: String): SavedModelInferenceModel {
        bundle = SavedModelBundle.load(pathToModel, "serve")
        session = bundle.session()
        val graph = bundle.graph()
        val graphDef = graph.toGraphDef()
        kGraph = KGraph(graphDef)
        return this
    }

    override fun close() {
        session.close()
        bundle.close()
        kGraph.close()
    }

    fun reshape2(function: (DoubleArray) -> Tensor<*>?) {
        reshape2 = function
    }
}

fun prepareModelForInference(init: SavedModelInferenceModel.() -> Unit): SavedModelInferenceModel =
    SavedModelInferenceModel()
        .apply(init)