/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/**
 * What's about Functional API usage in KotlinDL directly?
 *
 * Describe the model like the function of functions, where each layer is just a callable function.
 *
 * Combine two functions in special merge layers like Add or Concatenate.
 *
 * NOTE: Functional API supports one output and one input for model.
 */
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000

// TODO: move to tests
private val model = Functional.of(
    input,
    conv2D_2(conv2D_1),
    maxPool2D(conv2D_2),
    globalAvgPool2D(conv2D_8),
    dense_1(globalAvgPool2D),
    conv2D_1(input),
    conv2D_4(maxPool2D),
    conv2D_5(conv2D_4),
    add(conv2D_5, maxPool2D),
    conv2D_6(add),
    conv2D_7(conv2D_6),
    add_1(conv2D_7, add),
    conv2D_8(add_1),
    dense_2(dense_1)
)

fun main() {
    val (train, test) = fashionMnist()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.init()
        var accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after: $accuracy")
    }
}

