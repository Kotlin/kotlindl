/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.summary.EmptySummary
import org.jetbrains.kotlinx.dl.api.summary.ModelHubModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelWithSummary

/**
 * Base class for image classification models.
 * @property [internalModel] model used for prediction
 * @property [modelKindDescription] High-level description of the model. Used for model summary printing.
 * For the models from ModelHub it equals to the string representation of the [org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType].
 */
public abstract class ImageRecognitionModelBase<I>(
    protected val internalModel: InferenceModel,
    protected val modelKindDescription: String? = null
) : InferenceModel by internalModel, ModelWithSummary {
    /**
     * Preprocessing operation specific to this model.
     */
    protected abstract val preprocessing: Operation<I, FloatData>

    /**
     * Class labels from the dataset used for training.
     */
    protected abstract val classLabels: Map<Int, String>

    /**
     * Predicts object for the given [image].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [image] Input image.
     * @see preprocessing
     *
     * @return The label of the recognized object with the highest probability.
     */
    public fun predictObject(image: I): String {
        val (input, _) = preprocessing.apply(image)
        return classLabels[internalModel.predict(input)]!!
    }

    /**
     * Predicts [topK] objects for the given [image].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [image] Input image.
     * @param [topK] Number of top ranked predictions to return
     *
     * @see preprocessing
     *
     * @return The list of pairs <label, probability> sorted from the most probable to the lowest probable.
     */
    public fun predictTopKObjects(image: I, topK: Int = 5): List<Pair<String, Float>> {
        val (input, _) = preprocessing.apply(image)
        return internalModel.predictTopNLabels(input, classLabels, topK)
    }

    override fun summary(): ModelSummary {
        return if (internalModel is ModelWithSummary) {
            ModelHubModelSummary(internalModel.summary(), modelKindDescription)
        } else {
            ModelHubModelSummary(EmptySummary(), modelKindDescription)
        }
    }
}
