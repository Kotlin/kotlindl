package examples.experimental.hdf5


import api.keras.dataset.ImageDataset
import api.keras.layers.twodim.Conv2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import com.beust.klaxon.Klaxon
import examples.experimental.hdf5.lenetconfig.SequentialConfig
import examples.keras.fashionmnist.util.*
import examples.keras.mnist.util.AMOUNT_OF_CLASSES
import java.io.File

private

fun main() {
    val pathToConfig = "models/mnist/lenet/model_with_glorot_normal_init.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)

    val sequentialConfig = Klaxon()
        .parse<SequentialConfig>(jsonString)

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
        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = true
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }

}




