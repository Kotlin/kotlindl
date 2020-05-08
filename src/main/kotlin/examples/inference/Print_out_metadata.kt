package examples.inference

import tf_api.TensorFlow
import tf_api.inference.InferenceTFModel
import util.MnistUtils
import java.util.*

private const val IMAGE_PATH = "src/main/resources/datasets/test/t10k-images-idx3-ubyte"
private const val LABEL_PATH = "src/main/resources/datasets/test/t10k-labels-idx1-ubyte"
private const val PATH_TO_MODEL_1 = "src/main/resources/model3/saved_model.pb"
private const val PATH_TO_MODEL_2 = "src/main/resources/model1"

fun main() {
    println(TensorFlow.version())

    val images = MnistUtils.mnistAsList(
        IMAGE_PATH,
        LABEL_PATH, Random(0), 10000
    )

    /*val model1: TFModel = TFModel().loadGraph(PATH_TO_MODEL_1)

    println(model1.toString())*/

    val model2: InferenceTFModel = InferenceTFModel()
        .loadModel(PATH_TO_MODEL_2)

    println(model2.toString())
}