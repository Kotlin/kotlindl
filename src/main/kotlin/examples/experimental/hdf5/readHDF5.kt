package examples.experimental.hdf5


import api.inference.keras.loadConfig
import api.inference.keras.loadWeights
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import examples.keras.fashionmnist.util.*
import examples.keras.mnist.util.AMOUNT_OF_CLASSES
import examples.production.drawActivations
import examples.production.drawFilters
import examples.production.getLabel
import io.jhdf.HdfFile
import java.io.File

private

fun main() {
    val pathToConfig = "models/mnist/lenet/model.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)
    /* hdfFile.use { hdfFile ->
         recursivePrintGroup(hdfFile, hdfFile, 0)
     }*/

    val model = loadConfig<Float>(jsonConfigFile)
    model.compile(
        optimizer = Adam(),
        loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )
    model.summary()


    val pathToWeights = "models/mnist/lenet/mnist_weights_only1.h5"
    val realPathToWeights = ImageDataset::class.java.classLoader.getResource(pathToWeights).path.toString()

    val file = File(realPathToWeights)
    println(file.isFile)

    val hdfFile = HdfFile(file)

    model.loadWeights(hdfFile)

    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    model.use {
        //it.init()

        println(it.kGraph)
        var accuracy = 0.0
        val amountOfTestSet = 100


        val (result, activations) = it.predictAndGetActivations(train.getImage(11))
        println(activations.toString())
        drawActivations(activations)

        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getImage(imageId))

            if (prediction == getLabel(train, imageId))
                accuracy += (1.0 / amountOfTestSet)


            //println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId)}")
        }
        println("Accuracy: $accuracy")

        var weights = it.layers[0].getWeights() // first conv2d layer

        drawFilters(weights[0], 5.0)

        var weights2 = it.layers[2].getWeights() // second conv2d layer

        drawFilters(weights2[0], 10.0)

        var weights8 = it.layers[6].getWeights()

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")
    }
}


