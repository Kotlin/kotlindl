package examples.inference.keras.demo

import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.keras.loadKerasModel
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
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        it.init()

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy training $accuracy")
    }
}





