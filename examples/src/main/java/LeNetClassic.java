/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

import kotlin.Pair;
import org.jetbrains.kotlinx.dl.api.core.Sequential;
import org.jetbrains.kotlinx.dl.api.core.activation.Activations;
import org.jetbrains.kotlinx.dl.api.core.callback.Callback;
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant;
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal;
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros;
import org.jetbrains.kotlinx.dl.api.core.layer.Layer;
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D;
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding;
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense;
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input;
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D;
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten;
import org.jetbrains.kotlinx.dl.api.core.loss.SoftmaxCrossEntropyWithLogits;
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics;
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam;
import org.jetbrains.kotlinx.dl.api.core.optimizer.NoClipGradient;
import org.jetbrains.kotlinx.dl.api.core.regularizer.L1;
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2;
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset;
import org.jetbrains.kotlinx.dl.dataset.embedded.EmbeddedDatasetsKt;
import org.jetbrains.kotlinx.dl.dataset.embedded.MnistUtilKt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.jetbrains.kotlinx.dl.impl.summary.SummaryExKt.logSummary;

/**
 * This example demonstrates the ability to define and train LeNet-5 model in Java.
 */
public class LeNetClassic {
    private static final Logger logger = LoggerFactory.getLogger(LeNetClassic.class);

    public static final Integer EPOCHS = 2;
    public static final Integer TRAINING_BATCH_SIZE = 1000;
    public static final Long NUM_CHANNELS = 1L;
    public static final Long IMAGE_SIZE = 28L;
    public static final Long SEED = 12L;
    public static final Integer TEST_BATCH_SIZE = 1000;

    public static void main(String[] args) {
        Pair<OnHeapDataset, OnHeapDataset> result = EmbeddedDatasetsKt.mnist(new File("cache"));

        OnHeapDataset train = result.component1();
        OnHeapDataset test = result.component2();

        List<Layer> layers = new ArrayList<>();
        layers.add(new Input(new long[]{IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}, "x"));
        layers.add(new Conv2D(6, new int[]{5, 5}, new int[]{1, 1, 1, 1}, new int[]{1, 1, 1, 1}, Activations.Tanh, new GlorotNormal(SEED), new Zeros(), null, null, null, ConvPadding.SAME, true, "conv2d_1"));
        layers.add(new MaxPool2D(new int[]{1, 2, 2, 1}, new int[]{1, 2, 2, 1}, ConvPadding.VALID, "maxPool_1"));
        layers.add(new Conv2D(16, new int[]{5, 5}, new int[]{1, 1, 1, 1}, new int[]{1, 1, 1, 1}, Activations.Tanh, new GlorotNormal(SEED), new Zeros(), null, null, null, ConvPadding.SAME, true, "conv2d_2"));
        layers.add(new MaxPool2D(new int[]{1, 2, 2, 1}, new int[]{1, 2, 2, 1}, ConvPadding.VALID, "maxPool_2"));
        layers.add(new Flatten()); // 3136
        layers.add(new Dense(120, Activations.Tanh, new GlorotNormal(SEED), new Constant(0.1f), null, null, null, true, "dense_1"));
        layers.add(new Dense(84, Activations.Tanh, new GlorotNormal(SEED), new Constant(0.1f), null, null, null, true, "dense_2"));
        layers.add(new Dense(MnistUtilKt.NUMBER_OF_CLASSES, Activations.Linear, new GlorotNormal(SEED), new Constant(0.1f), new L2(0.001f), new L1(0.0001f), null, true, "dense_3"));

        try (Sequential lenet5Classic = Sequential.of(layers, false)) {
            Adam adam = new Adam(0.001f, 0.9f, 0.999f, 1e-07f, false, new NoClipGradient());
            lenet5Classic.compile(adam, new SoftmaxCrossEntropyWithLogits(), Metrics.ACCURACY);
            logSummary(lenet5Classic, logger);
            lenet5Classic.fit(train, EPOCHS, TRAINING_BATCH_SIZE, new Callback());

            Double accuracy = lenet5Classic.evaluate(test, TEST_BATCH_SIZE, new Callback()).getMetrics().get(Metrics.ACCURACY);
            System.out.println("Accuracy: " + accuracy);
        }
    }
}


