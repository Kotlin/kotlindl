package api.inference.savedmodel

import api.core.KGraph
import api.core.metric.Metrics
import api.inference.InferenceModel
import datasets.Dataset
import org.tensorflow.SavedModelBundle

/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class SavedModel : InferenceModel() {
    /** SavedModelBundle.*/
    private lateinit var bundle: SavedModelBundle

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): SavedModel {
            val model = SavedModel()

            model.bundle = SavedModelBundle.load(pathToModel, "serve")
            model.session = model.bundle.session()
            val graph = model.bundle.graph()
            val graphDef = graph.toGraphDef()
            model.kGraph = KGraph(graphDef)
            return model
        }
    }

    override fun predict(inputData: FloatArray): Int {
        require(reshapeFunction != null) { "Reshape functions is missed!" }

        reshapeFunction(inputData).use { tensor ->
            val runner = session.runner()
            return runner.feed(input.tfName, tensor)
                .fetch(output.tfName)
                .run()[0]
                .copyTo(LongArray(1))[0].toInt()
        }
    }

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @return Predicted class index.
     */
    public fun predict(inputData: FloatArray, inputTensorName: String, outputTensorName: String): Int {
        require(reshapeFunction != null) { "Reshape functions is missed!" }

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
    public fun predictAll(dataset: Dataset): List<Int> {
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
    public fun evaluate(
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

    override fun close() {
        session.close()
        bundle.close()
        kGraph.close()
    }
}

/*public fun prepareModelForInference(init: SavedModel.() -> Unit): SavedModel =
    SavedModel()
        .apply(init)*/