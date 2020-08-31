package examples.inference.keras.demo


import api.inference.keras.buildModelByJSONConfig
import api.inference.keras.loadWeights
import api.keras.Sequential
import api.keras.dataset.Dataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import examples.inference.keras.vgg.loadImageAndConvertToFloatArray
import io.jhdf.HdfFile
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)
    val model = buildModelByJSONConfig<Float>(jsonConfigFile)

    val imageNetClassLabels = prepareHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        println(it.kGraph)

        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\weights.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)

        val kernelDataPathTemplate = "/%s/%s_1/kernel:0"
        val biasDataPathTemplate = "/%s/%s_1/bias:0"
        it.loadWeights(hdfFile, kernelDataPathTemplate, biasDataPathTemplate)

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

fun predictTop5Labels(
    it: Sequential<Float>,
    floatArray: FloatArray,
    imageNetClassLabels: MutableMap<Int, String>
): MutableMap<Int, Pair<String, Float>> {
    val predictionVector = it.predictSoftly(floatArray, "Softmax").toMutableList()
    val predictionVector2 = it.predictSoftly(floatArray, "Softmax").toMutableList() // get copy of previous vector

    val top5: MutableMap<Int, Pair<String, Float>> = mutableMapOf()
    for (j in 1..5) {
        val max = predictionVector2.maxOrNull()
        val indexOfElem = predictionVector.indexOf(max!!)
        top5[j] = Pair(imageNetClassLabels[indexOfElem]!!, predictionVector[indexOfElem])
        predictionVector2.remove(max)
    }

    return top5
}

fun prepareHumanReadableClassLabels(): MutableMap<Int, String> {
    val pathToIndices = "/datasets/vgg/imagenet_class_index.json"

    fun parse(name: String): Any? {
        val cls = Parser::class.java
        return cls.getResourceAsStream(name)?.let { inputStream ->
            return Parser.default().parse(inputStream, Charsets.UTF_8)
        }
    }

    val classIndices = parse(pathToIndices) as JsonObject

    val imageNetClassIndices = mutableMapOf<Int, String>()

    for (key in classIndices.keys) {
        imageNetClassIndices[key.toInt()] = (classIndices[key] as JsonArray<*>)[1].toString()
    }
    return imageNetClassIndices
}




