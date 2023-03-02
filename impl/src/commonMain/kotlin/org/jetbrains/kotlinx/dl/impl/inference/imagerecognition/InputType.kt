/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * Different Neural Networks were trained on ImageNet dataset with different image preprocessing.
 * The main types of preprocessing widely used in `keras.applications` are presented in this enumeration.
 */
public enum class InputType {
    /**
     * This preprocessing will convert the images from RGB to BGR,
     * then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
     */
    CAFFE {
        public override fun preprocessing(channelsLast: Boolean): Operation<FloatData, FloatData> {
            return caffeStylePreprocessing(channelsLast)
        }
    },

    /**
     * This preprocessing will scale pixels between -1 and 1, sample-wise.
     */
    TF {
        public override fun preprocessing(channelsLast: Boolean): Operation<FloatData, FloatData> {
            return TfStylePreprocessing()
        }
    },

    /**
     * This preprocessing will scale pixels between 0 and 1,
     * then will normalize each channel with respect to the ImageNet dataset.
     */
    TORCH {
        public override fun preprocessing(channelsLast: Boolean): Operation<FloatData, FloatData> {
            return torchStylePreprocessing(channelsLast)
        }
    };

    /**
     * Returns preprocessing [Operation] corresponding to this preprocessing type.
     * @param [channelsLast] reflects whether channel dimension is the first or the last.
     */
    public abstract fun preprocessing(channelsLast: Boolean = true): Operation<FloatData, FloatData>
}
