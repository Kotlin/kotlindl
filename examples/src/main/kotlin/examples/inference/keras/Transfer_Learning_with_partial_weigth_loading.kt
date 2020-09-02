package examples.inference.keras

import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.dataset.Dataset
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*
import io.jhdf.HdfFile
import java.io.File

/**
 * Conv2D layers' weights are loaded from ImageNet, Dense weights are initialized by loaded initializers.
 */
fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
    val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
    val file = File(realPathToWeights)
    val hdfFile = HdfFile(file)

    val pathToConfig = "models/mnist/lenet/lenetMdl.json"
    val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)

    val model = buildModelByJSONConfig(jsonConfigFile)

    model.use {
        // Freeze conv2d layers, keep dense layers trainable
        val layerList = mutableListOf<Layer>()

        for (layer in it.layers) {
            if (layer::class == Conv2D::class) {
                layer.isTrainable = false
                layerList.add(layer)
            }
        }

        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadWeights(hdfFile, layerList)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = false,
            isWeightsInitRequired = false // for transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}




