package examples.inference.keras.demo

import LeNetClassic.SEED
import api.inference.keras.buildLayersByKerasJSONConfig
import api.inference.keras.loadWeightsForFrozenLayers
import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.Dataset
import api.keras.initializers.HeNormal
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.MaxPool2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*

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
    val (otherLayers, input) = buildLayersByKerasJSONConfig<Float>(jsonConfigFile)

    val layers = mutableListOf<Layer<Float>>()
    for (layer in otherLayers) {
        if (layer::class == Conv2D::class || layer::class == MaxPool2D::class) { // TODO
            layer.isTrainable = false // TODO: change for lambda
            layers.add(layer)
        }
    }

    layers.add(Flatten("new_flatten"))
    layers.add(
        Dense(
            name = "new_dense_1",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 256,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_2",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 128,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_3",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 64,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_4",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 10,
            activation = Activations.Linear
        )
    )
    val model = Sequential.of(input, layers)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = getWeightsFile()
        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false // for transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}





