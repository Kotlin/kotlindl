package tf_api

import org.tensorflow.op.Ops
import tensorflow.training.util.ImageDataset
import tf_api.blocks.LossFunction
import tf_api.blocks.Metric
import tf_api.blocks.Optimizer

class TFModel : InferenceTFModel() {

    /**
     * @optimizer — This is how the model is updated based on the data it sees and its loss function.
     * @loss — This measures how accurate the model is during training.
     * @metric — Used to monitor the training and testing steps.
     */
    fun compile(
        optimizer: Optimizer = Optimizer.ADAM,
        loss: LossFunction = LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        metric: Metric = Metric.ACCURACY
    ) {

    }

    fun fit(tf: Ops?, trainDataset: ImageDataset, epochs: Int, batchSize: Int) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

}