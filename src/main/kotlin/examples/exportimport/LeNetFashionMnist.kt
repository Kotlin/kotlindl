package examples.exportimport

import examples.keras.mnist.util.*
import tf_api.KGraph
import tf_api.keras.Sequential
import tf_api.keras.activations.Activations
import tf_api.keras.dataset.ImageDataset
import tf_api.keras.initializers.Xavier
import tf_api.keras.layers.Dense
import tf_api.keras.layers.Flatten
import tf_api.keras.layers.Input
import tf_api.keras.layers.twodim.Conv2D
import tf_api.keras.layers.twodim.ConvPadding
import tf_api.keras.layers.twodim.MaxPool2D
import tf_api.keras.loss.LossFunctions
import tf_api.keras.metric.Metrics
import tf_api.keras.optimizers.SGD
import java.io.File

private const val LEARNING_RATE = 0.0025f
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

/**
 * Kotlin implementation of LeNet on Keras.
 * Architecture could be copied here: https://github.com/TaavishThaman/LeNet-5-with-Keras/blob/master/lenet_5.py
 */
private val model = Sequential.of<Float>(
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
        kernelInitializer = Xavier(SEED),
        biasInitializer = Xavier(SEED),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = Xavier(SEED),
        biasInitializer = Xavier(SEED),
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = Xavier(SEED),
        biasInitializer = Xavier(SEED)
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = Xavier(SEED),
        biasInitializer = Xavier(SEED)
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear, // TODO: https://stats.stackexchange.com/questions/348036/difference-between-mathematical-and-tensorflow-implementation-of-softmax-crossen
        kernelInitializer = Xavier(SEED),
        biasInitializer = Xavier(SEED)
    )
)

fun main() {
    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        NUM_LABELS,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    val imageId = 1
    val trainImageLabel = train.getImageLabel(imageId)

    val maxIdx = trainImageLabel.indexOf(trainImageLabel.max()!!)

    println("Ground Truth: $maxIdx")


    model.use {
        val learningSchedule = mapOf(
            1 to 0.1f,
            2 to 0.05f,
            3 to 0.025f,
            4 to 0.01f,
            5 to 0.005f,
            6 to 0.0025f,
            7 to 0.001f
            /*8 to 0.001f,
            9 to 0.0005f,
            10 to 0.00025f*/
            /*11 to 0.0005f,
            12 to 0.0004f,
            13 to 0.0003f,
            14 to 0.0002f,
            15 to 0.0001f*/
        )

        it.compile(optimizer = SGD(learningSchedule), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val accuracy = it.evaluate(dataset = test, metric = Metrics.ACCURACY, batchSize = TEST_BATCH_SIZE)

        println("Accuracy $accuracy")

        val kGraph = (it as Sequential).getGraph()
        //println(kGraph)

        println("--------------Save graph to file----------------")
        File("lenetTFGraph").writeBytes(kGraph.tfGraph.toGraphDef())

        it.kGraph = KGraph(File("lenetTFGraph").readBytes(), "INFERENCE")

        //println(it.kGraph)

        val prediction = it.predict(train.getImage(imageId))

        println("Prediction: $prediction")

        val prediction2 = it.predict(train.getImage(imageId - 1))

        println("Prediction: $prediction2") // should be 9, ankle boot


        /*
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

         */

    }
}