package api.inference.savedmodel

import api.core.KGraph
import api.core.metric.Metrics
import api.inference.InferenceModel
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import util.MnistUtils

/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
open class SavedModel : InferenceModel() {
    /** SavedModelBundle.*/
    private lateinit var bundle: SavedModelBundle

    /** Reshape function. */
    private lateinit var reshape2: (DoubleArray) -> Tensor<*>?

    /**
     * Predicts the probabilities to be from known class for multi-classification task.
     */
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

    /**
     * Predicts labels for all [images].
     *
     * @param [images] Given list of images.
     */
    fun predictAll(images: List<MnistUtils.Image>): List<Double> {
        val predictedLabels: MutableList<Double> = mutableListOf()

        for (image in images) {
            val predictedLabel = predictOnImage(image)
            predictedLabels.add(predictedLabel[0].toDouble())
        }

        return predictedLabels
    }

    /**
     * Evaluates [testImages] via [metric].
     */
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

    /**
     * Loads model from SavedModelBundle format.
     */
    fun loadModel(pathToModel: String): SavedModel {
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

    /** Reshapes [DoubleArray] to [Tensor]. */
    fun reshape2(function: (DoubleArray) -> Tensor<*>?) {
        reshape2 = function
    }
}

/** Defines receiver for [SavedModel]. */
fun prepareModelForInference(init: SavedModel.() -> Unit): SavedModel =
    SavedModel()
        .apply(init)