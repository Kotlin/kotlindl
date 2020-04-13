package tf_api

import org.tensorflow.Graph
import tensorflow.training.util.ImageDataset
import tf_api.blocks.Metric
import tf_api.blocks.loss.LossFunctions
import tf_api.blocks.optimizers.Optimizers

abstract class TFModel<T : Number> : InferenceTFModel() {

    /**
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    abstract fun compile(
        optimizer: Optimizers = Optimizers.ADAM,
        loss: LossFunctions = LossFunctions.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        metric: Metric = Metric.ACCURACY
    )

    abstract fun fit(tf: Graph, trainDataset: ImageDataset, epochs: Int, batchSize: Int)
}