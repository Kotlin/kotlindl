/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.dataset

/**
 * This enum represents a type of COCO dataset labels.
 * Some models are trained on the COCO dataset with 80 classes (2014),
 * and some models are trained on the COCO dataset with 91 classes (2017).
 *
 * Also, sometimes labels are indexed from 1, and sometimes from 0.
 */
public enum class Coco {
    /**
     * COCO labels of version 2014 with 80 classes.
     */
    V2014,

    /**
     * COCO labels of version 2014 with 80 classes.
     */
    V2017;

    /**
     * Returns a map of COCO labels according to the [Coco] version.
     * @param [zeroIndexed] if true, then labels are indexed from 0, otherwise from 1.
     */
    public fun labels(zeroIndexed: Boolean = false): Map<Int, String> {
        return when (this) {
            V2014 -> if (zeroIndexed) toZeroIndexed(cocoCategories2014) else cocoCategories2014
            V2017 -> if (zeroIndexed) toZeroIndexed(cocoCategories2017) else cocoCategories2017
        }
    }

    private fun toZeroIndexed(labels: Map<Int, String>): Map<Int, String> {
        val zeroIndexedLabels = mutableMapOf<Int, String>()
        labels.forEach { (key, value) ->
            zeroIndexedLabels[key - 1] = value
        }
        return zeroIndexedLabels
    }
}

/**
 * 80 object categories in COCO dataset.
 *
 * Note that output class labels of SSD model do not correspond to ids in COCO annotations.
 * If you want to evaluate this model on the COCO validation/test set, you need to convert class predictions using appropriate mapping.
 *
 * @see <a href="https://cocodataset.org/#home">
 *     COCO dataset</a>
 */
public val cocoCategories2014: Map<Int, String> = mapOf(
    1 to "person",
    2 to "bicycle",
    3 to "car",
    4 to "motorbike",
    5 to "aeroplane",
    6 to "bus",
    7 to "train",
    8 to "truck",
    9 to "boat",
    10 to "traffic light",
    11 to "fire hydrant",
    12 to "stop sign",
    13 to "parking meter",
    14 to "bench",
    15 to "bird",
    16 to "cat",
    17 to "dog",
    18 to "horse",
    19 to "sheep",
    20 to "cow",
    21 to "elephant",
    22 to "bear",
    23 to "zebra",
    24 to "giraffe",
    25 to "backpack",
    26 to "umbrella",
    27 to "handbag",
    28 to "tie",
    29 to "suitcase",
    30 to "frisbee",
    31 to "skis",
    32 to "snowboard",
    33 to "sports ball",
    34 to "kite",
    35 to "baseball bat",
    36 to "baseball glove",
    37 to "skateboard",
    38 to "surfboard",
    39 to "tennis racket",
    40 to "bottle",
    41 to "wine glass",
    42 to "cup",
    43 to "fork",
    44 to "knife",
    45 to "spoon",
    46 to "bowl",
    47 to "banana",
    48 to "apple",
    49 to "sandwich",
    50 to "orange",
    51 to "broccoli",
    52 to "carrot",
    53 to "hot dog",
    54 to "pizza",
    55 to "donut",
    56 to "cake",
    57 to "chair",
    58 to "sofa",
    59 to "potted plant",
    60 to "bed",
    61 to "dining table",
    62 to "toilet",
    63 to "tvmonitor",
    64 to "laptop",
    65 to "mouse",
    66 to "remote",
    67 to "keyboard",
    68 to "cell phone",
    69 to "microwave",
    70 to "oven",
    71 to "toaster",
    72 to "sink",
    73 to "refrigerator",
    74 to "book",
    75 to "clock",
    76 to "vase",
    77 to "scissors",
    78 to "teddy bear",
    79 to "hair drier",
    80 to "toothbrush"
)


/**
 * 80 object categories in COCO dataset.
 *
 * @see <a href="https://cocodataset.org/#home">
 *     COCO dataset</a>
 */
public val cocoCategories2017: Map<Int, String> = mapOf(
    1 to "person",
    2 to "bicycle",
    3 to "car",
    4 to "motorcycle",
    5 to "airplane",
    6 to "bus",
    7 to "train",
    8 to "truck",
    9 to "boat",
    10 to "traffic light",
    11 to "fire hydrant",
    13 to "stop sign",
    14 to "parking meter",
    15 to "bench",
    16 to "bird",
    17 to "cat",
    18 to "dog",
    19 to "horse",
    20 to "sheep",
    21 to "cow",
    22 to "elephant",
    23 to "bear",
    24 to "zebra",
    25 to "giraffe",
    27 to "backpack",
    28 to "umbrella",
    31 to "handbag",
    32 to "tie",
    33 to "suitcase",
    34 to "frisbee",
    35 to "skis",
    36 to "snowboard",
    37 to "sports ball",
    38 to "kite",
    39 to "baseball bat",
    40 to "baseball glove",
    41 to "skateboard",
    42 to "surfboard",
    43 to "tennis racket",
    44 to "bottle",
    46 to "wine glass",
    47 to "cup",
    48 to "fork",
    49 to "knife",
    50 to "spoon",
    51 to "bowl",
    52 to "banana",
    53 to "apple",
    54 to "sandwich",
    55 to "orange",
    56 to "broccoli",
    57 to "carrot",
    58 to "hot dog",
    59 to "pizza",
    60 to "donut",
    61 to "cake",
    62 to "chair",
    63 to "couch",
    64 to "potted plant",
    65 to "bed",
    67 to "dining table",
    70 to "toilet",
    72 to "tv",
    73 to "laptop",
    74 to "mouse",
    75 to "remote",
    76 to "keyboard",
    77 to "cell phone",
    78 to "microwave",
    79 to "oven",
    80 to "toaster",
    81 to "sink",
    82 to "refrigerator",
    84 to "book",
    85 to "clock",
    86 to "vase",
    87 to "scissors",
    88 to "teddy bear",
    89 to "hair drier",
    90 to "toothbrush"
)
