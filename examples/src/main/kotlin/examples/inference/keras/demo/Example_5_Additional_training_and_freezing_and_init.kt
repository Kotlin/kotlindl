package examples.inference.keras.demo

import api.core.layer.Layer
import api.core.layer.twodim.Conv2D
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.keras.loadKerasModel
import api.inference.keras.loadWeights
import datasets.Dataset
import datasets.handlers.*

/** All weigths are loaded, the model just evaluated */
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


    val jsonConfigFile = getJSONConfigFile()
    val model = loadKerasModel(jsonConfigFile)

    model.use {
        val layerList = mutableListOf<Layer>()
        // Freeze conv2d layers, keep dense layers trainable
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

        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile, layerList)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false // for transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}





