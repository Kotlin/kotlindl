/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning

enum class InputType {
    /**
     * caffe: will convert the images from RGB to BGR,
    then will zero-center each color channel with
    respect to the ImageNet dataset,
    without scaling.
     */
    CAFFE,

    /**
     * will scale pixels between -1 and 1,
    sample-wise.
     */
    TF,

    /**
     * will scale pixels between 0 and 1 and then
    will normalize each channel with respect to the
    ImageNet dataset.
     */
    TORCH
}
