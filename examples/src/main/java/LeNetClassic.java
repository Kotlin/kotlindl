import api.keras.Sequential;
import api.keras.activations.Activations;
import api.keras.callbacks.Callback;
import api.keras.dataset.ImageDataset;
import api.keras.initializers.Constant;
import api.keras.initializers.GlorotNormal;
import api.keras.initializers.Zeros;
import api.keras.layers.Dense;
import api.keras.layers.Flatten;
import api.keras.layers.Input;
import api.keras.layers.twodim.Conv2D;
import api.keras.layers.twodim.ConvPadding;
import api.keras.layers.twodim.MaxPool2D;
import api.keras.loss.LossFunctions;
import api.keras.metric.Metrics;
import api.keras.optimizers.Adam;
import api.keras.optimizers.NoClipGradient;
import datasets.MnistUtilKt;

import static datasets.MnistUtilKt.*;

public class LeNetClassic {
    public static final Integer EPOCHS = 3;
    public static final Integer TRAINING_BATCH_SIZE = 1000;
    public static final Long NUM_CHANNELS = 1L;
    public static final Long IMAGE_SIZE = 28L;
    public static final Long SEED = 12L;
    public static final Integer TEST_BATCH_SIZE = 1000;

    public static final Sequential<Float> lenet5Classic = Sequential.Companion.of(
            new Input(new long[]{IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}, "x"),
            new Conv2D(6, new long[]{5, 5}, new long[]{1, 1, 1, 1}, new long[]{1, 1, 1, 1}, Activations.Tanh, new GlorotNormal(SEED), new Zeros(), ConvPadding.SAME, "conv2d_1"),
            new MaxPool2D(new int[]{1, 2, 2, 1}, new int[]{1, 2, 2, 1}, ConvPadding.VALID, "maxPool_1"),
            new Conv2D(16, new long[]{5, 5}, new long[]{1, 1, 1, 1}, new long[]{1, 1, 1, 1}, Activations.Tanh, new GlorotNormal(SEED), new Zeros(), ConvPadding.SAME, "conv2d_2"),
            new MaxPool2D(new int[]{1, 2, 2, 1}, new int[]{1, 2, 2, 1}, ConvPadding.VALID, "maxPool_2"),
            new Flatten(), // 3136
            new Dense(120, Activations.Tanh, new GlorotNormal(SEED), new Constant(0.1f), "dense_1"),
            new Dense(84, Activations.Tanh, new GlorotNormal(SEED), new Constant(0.1f), "dense_2"),
            new Dense(AMOUNT_OF_CLASSES, Activations.Linear, new GlorotNormal(SEED), new Constant(0.1f), "dense_3")
    );


    public static void main(String[] args) {
        var result = ImageDataset.Companion.createTrainAndTestDatasets(
                TRAIN_IMAGES_ARCHIVE,
                TRAIN_LABELS_ARCHIVE,
                TEST_IMAGES_ARCHIVE,
                TEST_LABELS_ARCHIVE,
                AMOUNT_OF_CLASSES,
                MnistUtilKt::extractImages,
                MnistUtilKt::extractLabels
        );

        var train = result.component1();
        var test = result.component2();

        var adam = new Adam<Float>(0.001f, 0.9f, 0.999f, 1e-07f, new NoClipGradient());

        try (lenet5Classic) {
            lenet5Classic.compile(adam, LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY, new Callback<>());
            lenet5Classic.summary(30, 26);
            lenet5Classic.fit(train, EPOCHS, TRAINING_BATCH_SIZE, true, true);

            var accuracy = lenet5Classic.evaluate(test, TEST_BATCH_SIZE).getMetrics().get(Metrics.ACCURACY);
            System.out.println("Accuracy: " + accuracy);
        }
    }
}


