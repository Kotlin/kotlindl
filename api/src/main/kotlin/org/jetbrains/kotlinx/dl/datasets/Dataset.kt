/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

public abstract class Dataset {
    /** Returns [BatchIterator] with fixed [batchSize]. */
    public abstract fun batchIterator(batchSize: Int): OnHeapDataset.BatchIterator

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    public abstract fun split(splitRatio: Double): Pair<OnHeapDataset, OnHeapDataset>

    /** Returns amount of data rows. */
    public abstract fun xSize(): Int

    /** Returns row by index [idx]. */
    public abstract fun getX(idx: Int): FloatArray

    /** Returns label as [FloatArray] by index [idx]. */
    public abstract fun getY(idx: Int): FloatArray

    /** Returns label as [Int] by index [idx]. */
    public abstract fun getLabel(idx: Int): Int
}
