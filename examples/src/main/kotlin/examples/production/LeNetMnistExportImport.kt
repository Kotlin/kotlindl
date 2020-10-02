package examples.production

import api.core.WrintingMode
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.InferenceModel
import datasets.Dataset
import datasets.handlers.*
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/lenet5"
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val (newTrain, validation) = train.split(0.95)

    val imageId1 = 0
    val imageId2 = 1
    val imageId3 = 2

    lenet5.use {
        it.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE,
            verbose = true
        )

        println(it.kGraph)

        it.save(File(PATH_TO_MODEL), wrintingMode = WrintingMode.OVERRIDE)

        val prediction = it.predict(train.getX(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getX(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getX(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")
    }


    val inferenceModel = InferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    inferenceModel.use {
        it.reshape(::mnistReshape)

        val prediction = it.predict(train.getX(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getX(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getX(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val pred = it.predict(train.getX(imageId))

            if (pred == getLabel(train, imageId))
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")
    }
}