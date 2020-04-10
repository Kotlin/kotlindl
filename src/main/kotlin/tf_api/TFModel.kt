package tf_api

import tf_api.blocks.Metric
import tf_api.blocks.Optimizer
import util.MnistUtils

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

    fun fit(trainImages: MutableList<MnistUtils.LabeledImage>) {

    }
}