Kotof is Keras API at the top of TensorFlow 1.15 and its Java API.

The implementation is inspired by https://github.com/dhruvrajan and its Java and Scala Keras APIs 
in https://github.com/dhruvrajan/tensorflow-keras-java and https://github.com/dhruvrajan/tensorflow-keras-scala.

Classic LeNet-5 with minor changes looks so Keras, isn't it?

```kotlin
private val model = Sequential.of<Float>(
    Input(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Zeros(),
        padding = ConvPadding.SAME
    ),
    AvgPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1)
    ),
    Flatten(), // 3136
    Dense(
        outputSize = 512,
        activation = Activations.Relu,
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear, // TODO: https://stats.stackexchange.com/questions/348036/difference-between-mathematical-and-tensorflow-implementation-of-softmax-crossen
        kernelInitializer = TruncatedNormal(SEED),
        biasInitializer = Constant(0.1f)
    )
)

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)
    val (train, test) = dataset.split(0.75)

    model.use {
        it.compile(optimizer = SGD(LEARNING_RATE), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.fit(trainDataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, isDebugMode = true)

        val accuracy = it.evaluate(testDataset = test, metric = Metrics.ACCURACY)

        println("Accuracy: $accuracy")
    }
}
```
