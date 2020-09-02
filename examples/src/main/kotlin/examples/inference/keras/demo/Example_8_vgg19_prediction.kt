package examples.inference.keras.demo


import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.dataset.Dataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import examples.inference.keras.vgg.loadImageAndConvertToFloatArray
import io.jhdf.HdfFile
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg19\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)
    val model = buildModelByJSONConfig(jsonConfigFile)

    val imageNetClassLabels = prepareHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg19\\hdf5\\weights.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = loadImageAndConvertToFloatArray(inputStream)

            val (res, _) = it.predictAndGetActivations(floatArray, "Softmax")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, floatArray, imageNetClassLabels)

            println(top5.toString())
        }
    }
}




