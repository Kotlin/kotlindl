package api.inference.savedmodel

import api.core.KGraph
import api.core.metric.Metrics
import api.inference.InferenceModel
import datasets.Dataset
import org.tensorflow.SavedModelBundle

/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
open class SavedModel : InferenceModel() {
    /** SavedModelBundle.*/
    private lateinit var bundle: SavedModelBundle

    override fun predict(inputData: FloatArray): Int {
        reshapeFunction(inputData).use { tensor ->
            val runner = session.runner()
            return runner.feed(input.tfName, tensor)
                .fetch(output.tfName)
                .run()[0]
                .copyTo(LongArray(1))[0].toInt()
        }
    }

    fun predict(inputData: FloatArray, inputTensorName: String, outputTensorName: String): Int {
        reshapeFunction(inputData).use { tensor ->
            val runner = session.runner()
            return runner.feed(inputTensorName, tensor)
                .fetch(outputTensorName)
                .run()[0]
                .copyTo(LongArray(1))[0].toInt()
        }
    }

    /**
     * Predicts labels for all [images].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [dataset] Dataset.
     */
    fun predictAll(dataset: Dataset): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i))
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    /**
     * Evaluates [dataset] via [metric].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     */
    fun evaluate(
        dataset: Dataset,
        metric: Metrics
    ): Double {

        return if (metric == Metrics.ACCURACY) {
            var counter = 0
            for (i in 0 until dataset.xSize()) {
                val predictedLabel = predict(dataset.getX(i))
                if (predictedLabel == dataset.getLabel(i))
                    counter++
            }

            (counter.toDouble() / dataset.xSize())
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
}

/** Defines receiver for [SavedModel]. */
fun prepareModelForInference(init: SavedModel.() -> Unit): SavedModel =
    SavedModel()
        .apply(init)