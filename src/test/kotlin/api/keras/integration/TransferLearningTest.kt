package api.keras.integration

import api.inference.keras.loadConfig
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.GlorotNormal
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.io.File

class TransferLearningTest : IntegrationTest() {
    @Test
    fun loadLeNetJSONConfig() {
        val pathToConfig = "models/mnist/lenet/model_with_glorot_normal_init.json"
        val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = loadConfig<Float>(jsonConfigFile)

        Assertions.assertEquals(testModel.layers.size, 8)
        Assertions.assertTrue(testModel.getLayer("conv2d_14") is Conv2D)
        Assertions.assertTrue((testModel.getLayer("conv2d_14") as Conv2D).kernelInitializer is GlorotNormal)
        Assertions.assertTrue((testModel.getLayer("conv2d_14") as Conv2D).biasInitializer is GlorotNormal)
        Assertions.assertTrue((testModel.getLayer("conv2d_14") as Conv2D).padding == ConvPadding.SAME)
        Assertions.assertTrue((testModel.getLayer("conv2d_14") as Conv2D).activation == Activations.Relu)
        Assertions.assertTrue(testModel.getLayer("conv2d_15") is Conv2D)
        Assertions.assertTrue(testModel.getLayer("conv2d_14").isTrainable)
        Assertions.assertTrue(testModel.getLayer("conv2d_15").hasActivation())
        Assertions.assertTrue(testModel.getLayer("flatten_3").isTrainable)
        Assertions.assertFalse(testModel.getLayer("flatten_3").hasActivation())
        Assertions.assertArrayEquals(testModel.firstLayer.packedDims, longArrayOf(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    }

    @Test
    fun loadLeNetJSONConfigAndTrain() {
        val pathToConfig = "models/mnist/lenet/model_with_glorot_normal_init.json"
        val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()
        val jsonConfigFile = File(realPathToConfig)

        val testModel = loadConfig<Float>(jsonConfigFile)

        val (train, test) = ImageDataset.createTrainAndTestDatasets(
            FASHION_TRAIN_IMAGES_ARCHIVE,
            FASHION_TRAIN_LABELS_ARCHIVE,
            FASHION_TEST_IMAGES_ARCHIVE,
            FASHION_TEST_LABELS_ARCHIVE,
            AMOUNT_OF_CLASSES,
            ::extractFashionImages,
            ::extractFashionLabels
        )

        testModel.use {
            it.compile(
                optimizer = Adam(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            it.fit(
                dataset = train,
                validationRate = VALIDATION_RATE,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = VALIDATION_BATCH_SIZE,
                verbose = false,
                isWeightsInitRequired = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = VALIDATION_BATCH_SIZE).metrics[Metrics.ACCURACY]

            Assertions.assertEquals(accuracy!!, 0.7888998985290527, EPS)
        }
    }
}