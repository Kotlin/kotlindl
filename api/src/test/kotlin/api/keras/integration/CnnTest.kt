package api.keras.integration

import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.initializers.Constant
import api.keras.initializers.HeNormal
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.Dataset
import datasets.handlers.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

private const val INCORRECT_AMOUNT_OF_CLASSES_1 = 11

internal class CnnTest : IntegrationTest() {
    private val testModel = Sequential.of(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        ),
        Conv2D(
            filters = 32,
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "conv2d_1"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_1"
        ),
        Conv2D(
            filters = 64,
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
            name = "conv2d_2"
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1),
            name = "maxPool_2"
        ),
        Flatten(name = "flatten_1"), // 3136
        Dense(
            outputSize = 512,
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_1"
        ),
        Dense(
            outputSize = AMOUNT_OF_CLASSES,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = Constant(0.1f),
            name = "dense_2"
        )
    )

    @Test
    fun mnistDatasetCreation() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        assertEquals(train.xSize(), 60000)
        assertEquals(test.xSize(), 10000)
    }


    @Test
    fun trainingLeNetModel() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModel.use {
            it.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            val trainingHistory =
                it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = false)

            assertEquals(trainingHistory.batchHistory.size, 60)
            assertEquals(trainingHistory.batchHistory[0].epoch, 1)
            assertEquals(trainingHistory.batchHistory[0].batch, 0)
            assertEquals(trainingHistory.batchHistory[0].lossValue, 2.9598662853240967, EPS)
            assertEquals(trainingHistory.batchHistory[0].metricValue, 0.09799999743700027, EPS)

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            assertEquals(accuracy!!, 0.9739, EPS)
        }
    }

    @Test
    fun incorrectAmountOfClassesInTheLastDenseLayer() {
        val testModelWithSmallAmountOfClasses = Sequential.of(
            Input(
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            ),
            Conv2D(
                filters = 32,
                kernelSize = longArrayOf(5, 5),
                strides = longArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                name = "conv2d_1"
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "maxPool_1"
            ),
            Conv2D(
                filters = 64,
                kernelSize = longArrayOf(5, 5),
                strides = longArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME,
                name = "conv2d_2"
            ),
            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1),
                strides = intArrayOf(1, 2, 2, 1),
                name = "maxPool_2"
            ),
            Flatten(name = "flatten_1"), // 3136
            Dense(
                outputSize = 512,
                activation = Activations.Relu,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f),
                name = "dense_1"
            ),
            Dense(
                outputSize = INCORRECT_AMOUNT_OF_CLASSES_1,
                activation = Activations.Linear,
                kernelInitializer = HeNormal(SEED),
                biasInitializer = Constant(0.1f),
                name = "dense_2"
            )
        )


        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        testModelWithSmallAmountOfClasses.use {
            it.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.fit(
                        dataset = train,
                        epochs = EPOCHS,
                        batchSize = TRAINING_BATCH_SIZE,
                        verbose = false
                    )
                }
            assertEquals(
                "The calculated [from the Sequential model] label batch shape [1000, 11] doesn't match actual data buffer size 10000. \n" +
                        "Please, check the input label data or correct amount of classes [amount of neurons] in last Dense layer, if you have a classification problem.\n" +
                        "Highly likely, you have different amount of classes presented in data and described in model as desired output.",
                exception.message
            )
        }
    }
}