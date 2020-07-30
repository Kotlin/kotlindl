package examples.production

import api.keras.Sequential
import api.keras.dataset.ImageDataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import examples.keras.mnist.util.*
import javax.swing.JFrame

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val imageId = 0

    lenet5.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.MAE,
            metric = Metrics.MAE
        )

        (it as Sequential).summary()

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE,
            verbose = true
        )

        var weights = it.layers[0].getWeights() // first conv2d layer

        drawFilters(weights[0])

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy $accuracy")

        val (prediction, activations) = it.predictAndGetActivations(train.getImage(imageId))

        println("Prediction: $prediction")

        drawActivations(activations)


        val trainImageLabel = train.getImageLabel(imageId)

        val maxIdx = trainImageLabel.indexOf(trainImageLabel.max()!!)

        println("Ground Truth: $maxIdx")
    }
}

fun drawActivations(activations: List<*>) {
    val frame = JFrame("Visualise the matrix weights on Relu")
    frame.contentPane.add(ReluGraphics(activations[0] as Array<Array<Array<FloatArray>>>))
    frame.setSize(1500, 1500)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false

    val frame2 = JFrame("Visualise the matrix weights on Relu_1")
    frame2.contentPane.add(ReluGraphics2(activations[1] as Array<Array<Array<FloatArray>>>))
    frame2.setSize(1500, 1500)
    frame2.isVisible = true
    frame2.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame2.isResizable = false
}

fun drawFilters(filters: Array<*>, colorCoefficient: Double = 2.0) {
    val frame = JFrame("Filters")
    frame.contentPane.add(Conv2dJPanel(filters as Array<Array<Array<FloatArray>>>, colorCoefficient))
    frame.setSize(1000, 1000)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}