/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.history

import java.util.*

/**
 * Contains all recorded batch events as a list of [BatchEvent] objects.
 *
 * NOTE: Used to record [BatchEvent] during prediction and evaluation phases.
 */
public class History {
    /** Plain list of batch events. */
    private val history: MutableList<BatchEvent> = mutableListOf()

    /** Indexed and sorted storage of batch events. */
    private val historyByBatch: TreeMap<Int, BatchEvent> = TreeMap()

    /**
     * Appends tracked data from one batch event.
     */
    public fun appendBatch(batch: Int, lossValue: Double, metricValues: List<Double>) {
        val newEvent = BatchEvent(batch, lossValue, metricValues)
        addNewBatchEvent(newEvent, batch)
    }

    /**
     * Appends one [BatchEvent].
     */
    public fun appendBatch(batchEvent: BatchEvent) {
        addNewBatchEvent(batchEvent, batchEvent.batchIndex)
    }

    private fun addNewBatchEvent(batchEvent: BatchEvent, batchIndex: Int) {
        history.add(batchEvent)
        historyByBatch[batchIndex] = batchEvent
    }

    /**
     * Returns last [BatchEvent]
     */
    public fun lastBatchEvent(): BatchEvent {
        return historyByBatch.lastEntry().value
    }
}

/**
 * One record in [History] objects containing tracked data from one batch.
 *
 * @constructor Creates [BatchEvent] from [batchIndex], [lossValue], [metricValues].
 * @param batchIndex Batch index.
 * @param lossValue Final value of loss function.
 * @param metricValues Final values of chosen metrics.
 */
public class BatchEvent(public val batchIndex: Int, public val lossValue: Double, public val metricValues: List<Double>) {
    override fun toString(): String {
        return "BatchEvent(batchIndex=$batchIndex, lossValue=$lossValue, metricValues=$metricValues)"
    }
}
