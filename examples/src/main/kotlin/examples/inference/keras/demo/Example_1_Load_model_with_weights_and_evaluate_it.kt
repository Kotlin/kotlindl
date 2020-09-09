package examples.inference.keras.demo

import api.inference.keras.loadKerasModel
import api.inference.keras.loadWeights
import api.keras.dataset.Dataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*
import io.jhdf.HdfFile
import java.io.File

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
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile)

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy training $accuracy")
    }
}

fun getJSONConfigFile(): File {
    /*val pathToConfig = "models/mnist/lenet/lenetMdl.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    return File(realPathToConfig)*/
    return File("C:\\zaleslaw\\home\\models\\demo\\lenet\\modelConfig.json")
}

fun getWeightsFile(): HdfFile {
    /* val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
     val realPathToWeights = ImageDataset::class.java.classLoader.getResource(pathToWeights).path.toString()
     val file = File(realPathToWeights)*/
    val file = File("C:\\zaleslaw\\home\\models\\demo\\lenet\\mnist_weights_only.h5")
    return HdfFile(file)
}




