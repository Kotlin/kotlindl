Transfer learning is a popular deep learning technique that allows you to save time on training a neural network while achieving great performance too. 
It leverages existing pre-trained models and allows you to tweak them to your task by training only a part of the neural network.

In this tutorial, we will take a pre-trained ResNet'50 Keras model that has been trained on a large dataset (ImageNet) to classify color images into 1000 categories. 
We will load this model with KotlinDL, and fine-tune it by training only some of its layers to classify images from our own dataset.

## Data preparation
For the purposes of this tutorial, we downloaded a dataset containing images of dogs and cats 
(it's a subset of the dataset from the famous Kaggle competition with only 50 images per class instead of 12500 images per class like the original dataset). 
Next, we resized the images to be 224 x 224 pixels. This is the same image size as ResNet'50 was trained on. 
It is important for all the images to be the same size, however, 
it is not critical for them to be of the exact size as what the model was trained on – if needed, we can replace the input layer.

We've stored the image dataset so each class is in its own folder: 
```
small-dogs-vs-cats/
    cat/
    dogs/
```
There are, of course, many ways to organize your own dataset. 
This way makes it easier to get the labels for all the examples based on the folder names 
(we assume that the folder with the name _cat_ contains images of cats).

Now we need to create a `Dataset` from these images. 
You can do so via the Image Preprocessing Pipeline description, and building a dataset from those. 

**Note**: The preprocessing DSL has changed in KotlinDL 0.5.0.
You can find the docs for the previous version of the DSL [here](https://github.com/Kotlin/kotlindl/blob/release_0.4/docs/transfer_learning.md).

Here's code that will go through a folder structure received via ```dogsCatsSmallDatasetPath()```, loads and resizes the images, and applies the ResNet'50 specific preprocessing.

```kotlin
val preprocessing = pipeline<BufferedImage>()
    .resize {
        outputHeight = 224
        outputWidth = 224
        interpolation = InterpolationType.BILINEAR
    }
    .convert { colorMode = ColorMode.RGB }
    .toFloatArray { }
    .call(TFModels.CV.ResNet50().preprocessor)

val dogsVsCatsDatasetPath = dogsCatsSmallDatasetPath()
val dataset = OnFlyImageDataset.create(
    File(dogsVsCatsDatasetPath),
    FromFolders(mapping = mapOf("cat" to 0, "dog" to 1)),
    preprocessing
).shuffle()

val (train, test) = dataset.split(0.7)
```  
In the final lines, after creating a dataset, we shuffle the data, so that when we split it into training and testing portions, we do not get a test set containing only images of one class.    
 
## ResNet'50
KotlinDL bundles a lot of pre-trained models available via ModelHub object. 
You can either train a model from scratch yourself and store it for later use on other tasks, or you can import a pre-trained Keras model with compatible architecture.  

In this tutorial, we will load ResNet'50 model and weights that are made available in the ModelHub: 

```kotlin
val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
val modelType = TFModels.CV.ResNet50()
val model = modelHub.loadModel(modelType)
```

## Transfer Learning
Now we have created the dataset, and we have a model, we can put everything together, and apply the transfer learning technique.

At this point, we need to decide which layers of this model we want to fine-tune, which ones we want to leave as is and if we want to add or remove layers. 
You can use `model.logSummary()` to inspect the model's architecture.

This model consists mainly of Conv2D and MaxPool2D layers and has a couple of dense layers at the end. One way to do transfer learning (although, of course, not the only one) is to leave the convolutional layers as they are, and re-train the dense layers. 
So this is what we will do:
- We'll "freeze" all the Conv2D and MaxPool2D layers – the weights for them will be loaded, but they will not be trained any further.
- The last layer of the original model classifies 1000 classes, but we only have two, so we'll dispose of it, and add another final prediction layer (and one intermediate dense layer to achieve better accuracy).   

```kotlin
val layers = mutableListOf<Layer>()

for (layer in model.layers) {
    layer.isTrainable = false
    layers.add(layer)
}

val lastLayer = layers.last()
for (outboundLayer in lastLayer.inboundLayers)
    outboundLayer.outboundLayers.remove(lastLayer)

layers.removeLast()

val newDenseLayer = Dense(
    name = "new_dense_1",
    kernelInitializer = HeNormal(),
    biasInitializer = HeNormal(),
    outputSize = 64,
    activation = Activations.Relu
)
newDenseLayer.inboundLayers.add(layers.last())
layers.add(
    newDenseLayer
)

val newDenseLayer2 = Dense(
    name = "new_dense_2",
    kernelInitializer = HeNormal(),
    biasInitializer = HeNormal(),
    outputSize = 2,
    activation = Activations.Linear
)
newDenseLayer2.inboundLayers.add(layers.last())

layers.add(
    newDenseLayer2
)

val newModel = Functional.of(layers)
```

Finally, we can train this model. The only difference to training a model from scratch will be loading the weights for the frozen layers. 
These will not be further trained – that's how we can leverage the fact that this model has already learned some patterns on a much larger dataset.  

```kotlin
 newModel.use {
    it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    val hdfFile = modelHub.loadWeights(modelType)
    it.loadWeightsForFrozenLayers(hdfFile)

    val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]
    println("Accuracy before training $accuracyBeforeTraining")

    it.fit(
        dataset = train,
        batchSize = 8,
        epochs = 2
    )

    val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]

    println("Accuracy after training $accuracyAfterTraining")
}
```

That is it! Congratulations! You have learned how to use the transfer learning technique with KotlinDL and Functional API.  
