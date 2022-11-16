/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import kotlin.math.min

/** Just abstract Dataset. */
public abstract class Dataset {
    /** Splits datasets on two sub-datasets according [splitRatio].*/
    public abstract fun split(splitRatio: Double): Pair<Dataset, Dataset>

    /** Returns amount of data rows. */
    public abstract fun xSize(): Int

    /** Returns row by index [idx]. */
    public abstract fun getX(idx: Int): FloatArray

    /** Returns label as [Int] by index [idx]. */
    public abstract fun getY(idx: Int): Float

    /** Shuffles the dataset. */
    public abstract fun shuffle(): Dataset

    /**
     * An iterator over a [Dataset].
     */
    public inner class BatchIterator internal constructor(
        private val batchSize: Int
    ) : Iterator<DataBatch?> {

        private var batchStart = 0

        override fun hasNext(): Boolean = batchStart < xSize()

        override fun next(): DataBatch {
            val batchLength = min(batchSize, xSize() - batchStart)
            val batch = createDataBatch(batchStart, batchLength)
            batchStart += batchSize
            return batch
        }
    }

    /** Creates data batch that starts from [batchStart] with length [batchLength]. */
    protected abstract fun createDataBatch(batchStart: Int, batchLength: Int): DataBatch


    /** Returns [BatchIterator] with fixed [batchSize]. */
    public fun batchIterator(batchSize: Int): BatchIterator {
        return BatchIterator(batchSize)
    }
}
