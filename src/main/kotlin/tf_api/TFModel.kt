package tf_api

import org.tensorflow.Graph
import org.tensorflow.op.Ops
import tensorflow.training.util.ImageDataset
import tf_api.inference.InferenceTFModel
import tf_api.keras.Metric
import tf_api.keras.loss.LossFunctions
import tf_api.keras.optimizers.Optimizers

abstract class TFModel<T : Number> : InferenceTFModel() {

    /**
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    abstract fun compile(
        tf: Ops,
        optimizer: Optimizers = Optimizers.ADAM,
        loss: LossFunctions = LossFunctions.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        metric: Metric = Metric.ACCURACY
    )

    abstract fun fit(graph: Graph, tf: Ops, trainDataset: ImageDataset, epochs: Int, batchSize: Int)

    abstract fun evaluate(testDataset: ImageDataset, metric: Metric): Double
}