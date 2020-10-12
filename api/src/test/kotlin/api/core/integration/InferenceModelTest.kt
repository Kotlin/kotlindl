package api.core.integration

import api.core.SavingFormat
import api.core.Sequential
import api.core.WrintingMode
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.initializer.HeUniform
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.Losses
import api.core.metric.Accuracy
import api.core.metric.Metrics
import api.core.optimizer.SGD
import api.inference.InferenceModel
import datasets.Dataset
import datasets.handlers.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.tensorflow.Tensor
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Path

private const val EPS = 0.3
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 2000
private const val TEST_BATCH_SIZE = 1000
private const val AMOUNT_OF_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private val kernelInitializer = HeNormal(12L)
private val biasInitializer = HeUniform(12L)

class InferenceModelTest {
    private val lenet5Layers = listOf(
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

    private fun mnistReshape(image: FloatArray): Tensor<*> {
        val reshaped = Array(
            1
        ) { Array(28) { Array(28) { FloatArray(1) } } }
        for (i in image.indices) reshaped[0][i / 28][i % 28][0] = image[i]
        return Tensor.create(reshaped)
    }

    @Test
    fun basicInference(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE,
                verbose = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )
        }

        val inferenceModel = InferenceModel.load(tempDir.toFile(), loadOptimizerState = false)
        inferenceModel.use {
            it.reshape(::mnistReshape)
            var accuracy = 0.0
            val amountOfTestSet = 10000
            for (imageId in 0..amountOfTestSet) {
                val prediction = it.predict(train.getX(imageId))

                if (prediction == train.getLabel(imageId))
                    accuracy += (1.0 / amountOfTestSet)
            }

            assertTrue(accuracy > 0.5)
        }
    }

    @Test
    fun emptyInferenceModel() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        val inferenceModel = InferenceModel()
        inferenceModel.use {
            it.reshape(::mnistReshape)

            val exception =
                Assertions.assertThrows(IllegalStateException::class.java) {
                    it.predict(train.getX(0))
                }
            assertEquals(
                "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method.",
                exception.message
            )
        }
    }

    @Test
    fun missedReshapeFunction() {
        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        val inferenceModel = InferenceModel()
        inferenceModel.use {
            val exception =
                Assertions.assertThrows(IllegalArgumentException::class.java) {
                    it.predict(train.getX(0))
                }
            assertEquals(
                "Reshape functions is missed!",
                exception.message
            )
        }
    }

    @Test
    fun createInferenceModelOnJSONConfig(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE,
                verbose = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )
        }

        val exception =
            Assertions.assertThrows(FileNotFoundException::class.java) {
                InferenceModel.load(tempDir.toFile())
            }
        assertEquals(
            "File 'graph.pb' is not found. This file must be in the model directory. It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.TF_GRAPH.",
            exception.message
        )
    }

    @Test
            /** Manually removed file 'variableNames.txt'. */
    fun createInferenceModelOnCorruptedVariableData(@TempDir tempDir: Path?) {
        assertTrue(tempDir!!.toFile().isDirectory)

        val lenet5 = Sequential.of(lenet5Layers)

        val (train, test) = Dataset.createTrainAndTestDatasets(
            TRAIN_IMAGES_ARCHIVE,
            TRAIN_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE,
            TEST_LABELS_ARCHIVE,
            datasets.handlers.AMOUNT_OF_CLASSES,
            ::extractImages,
            ::extractLabels
        )

        lenet5.use {
            it.compile(
                optimizer = SGD(learningRate = 0.05f),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Accuracy()
            )

            it.fit(
                dataset = train,
                epochs = EPOCHS,
                batchSize = TRAINING_BATCH_SIZE,
                verbose = true
            )

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
            if (accuracy != null) {
                assertTrue(accuracy > 0.5)
            }

            it.save(
                modelDirectory = tempDir.toFile(),
                savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
                writingMode = WrintingMode.OVERRIDE
            )
        }

        File(tempDir.toFile().absolutePath + "/variableNames.txt").delete()

        val exception =
            Assertions.assertThrows(FileNotFoundException::class.java) {
                InferenceModel.load(tempDir.toFile())
            }
        assertEquals(
            "File 'variableNames.txt' is not found. This file must be in the model directory. It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES.",
            exception.message
        )
    }
}