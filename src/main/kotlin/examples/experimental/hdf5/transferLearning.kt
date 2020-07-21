package examples.experimental.hdf5


import api.keras.dataset.ImageDataset
import api.keras.layers.twodim.Conv2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import com.beust.klaxon.Klaxon
import examples.experimental.hdf5.lenetconfig.SequentialConfig
import examples.keras.mnist.util.*
import examples.production.drawFilters
import examples.production.getLabel
import io.jhdf.HdfFile
import java.io.File

private

fun main() {
    val pathToConfig = "models/mnist/lenet/model.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
    //println(jsonString)

    val sequentialConfig = Klaxon()
        .parse<SequentialConfig>(jsonString)

    //println(sequentialConfig.toString())

    val model = buildSequentialModelByJSONConfig(sequentialConfig!!)
    for (layer in model.layers) {
        println(layer.name)
        if (layer::class == Conv2D::class)
            layer.isTrainable = false
    }

    model.compile(
        optimizer = Adam<Float>(),
        loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )
    model.summary()


    val pathToWeights = "models/mnist/lenet/mnist_weights_only.h5"
    val realPathToWeights = ImageDataset::class.java.classLoader.getResource(pathToWeights).path.toString()

    val file = File(realPathToWeights)
    println(file.isFile)

    val hdfFile = HdfFile(file)

    loadWeightsToModel(model, hdfFile)

    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )



    model.use {
        for (imageId in 0..100) {
            val prediction = it.predict(test.getImage(imageId))

            println("Prediction: $prediction Ground Truth: ${getLabel(test, imageId)}")
        }


        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        var weights = it.layers[0].getWeights() // first conv2d layer

        drawFilters(weights[0])

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 2,
            trainBatchSize = 100,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false // for transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }

}




