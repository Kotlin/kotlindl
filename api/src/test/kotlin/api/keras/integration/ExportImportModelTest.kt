package api.keras.integration

import api.ModelWritingMode
import api.keras.ModelFormat
import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.Dataset
import api.keras.initializers.GlorotNormal
import api.keras.initializers.GlorotUniform
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.RMSProp
import api.keras.optimizers.SGD
import datasets.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

private const val EPS = 1e-2
private const val SEED = 12L
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000
private const val AMOUNT_OF_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private val kernelInitializer = GlorotNormal(SEED)
private val biasInitializer = GlorotUniform(SEED)

val lenet5 = Sequential.of(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS,
        name = "input_0"
    ),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_1"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_2"
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_3"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_4"
    ),
    Flatten(name = "flatten_5"), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_6"
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_7"
    ),
    Dense(
        outputSize = AMOUNT_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_8"
    )
)

class ExportImportModelTest {
    @Test
    fun exportImport(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        val (newTrain, validation) = train.split(0.95)

        lenet5.use {
            it.compile(optimizer = SGD(learningRate = 0.05f), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

            it.summary()

            it.fit(
                trainingDataset = newTrain,
                validationDataset = validation,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = true
            )

            it.save(
                pathToModelDirectory = tempDir.toString(),
                modelFormat = ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES,
                modelWritingMode = ModelWritingMode.OVERRIDE
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            assertEquals(0.7603000402450562, accuracy!!, EPS)
        }

        val model = Sequential.load(tempDir.toString())

        model.use {
            // Freeze conv2d layers, keep dense layers trainable
            for (layer in it.layers) {
                if (layer::class == Conv2D::class)
                    layer.isTrainable = false
            }

            it.compile(
                optimizer = RMSProp(),
                loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )
            it.summary()

            it.loadVariablesFromTxtFiles(tempDir.toString())

            val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.7603000402450562, accuracyBefore!!, EPS)

            it.fit(
                dataset = train,
                validationRate = 0.1,
                epochs = EPOCHS,
                trainBatchSize = TRAINING_BATCH_SIZE,
                validationBatchSize = TEST_BATCH_SIZE,
                verbose = false,
                isWeightsInitRequired = false // for transfer learning
            )

            val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
            assertEquals(0.8234000205993652, accuracyAfterTraining!!, EPS)
        }
    }
}