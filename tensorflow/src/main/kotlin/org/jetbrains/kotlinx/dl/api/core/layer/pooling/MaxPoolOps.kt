/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

/** TensorFlow op taking the pooling window and strides as attributes. */
private const val MAX_POOL_OP = "MaxPool"

/**
 * Builds a `MaxPool` node — the variant that takes the pooling window and strides as **attributes**
 * rather than as input tensors.
 *
 * [org.tensorflow.op.NnOps.maxPool] cannot be used for this. Its only overload in TensorFlow Java
 * takes `ksize` and `strides` as `Operand<TInt32>`, and so emits `MaxPoolV2`
 * (`org.tensorflow.op.nn.MaxPool.OP_NAME` is literally `"MaxPoolV2"`). The two ops compute the same
 * thing, but they are not equally well supported:
 *
 * * Apple's `tensorflow-metal` PluggableDevice registers kernels for `MaxPool` and `MaxPoolGrad`
 *   but **not** for `MaxPoolV2`/`MaxPoolGradV2`, so a `MaxPoolV2` node falls back to the CPU and
 *   drags the surrounding graph through a host round trip. On a 224x224 CNN training step that
 *   costs roughly a third of the available Metal speedup.
 * * More generally, static attributes let the shape inference and graph optimisers reason about the
 *   window, which input tensors do not.
 *
 * KotlinDL always knows the window and strides when the graph is built, so nothing is lost by using
 * the attribute form. The gradient is registered for `MaxPool` in TensorFlow's C++ gradient registry,
 * so pooling layers remain trainable.
 *
 * @param input rank-4 input, `(batch, height, width, channels)` for the default data format.
 * @param poolSize sliding window per input dimension.
 * @param strides stride per input dimension.
 * @param padding TensorFlow padding name, `"SAME"` or `"VALID"`.
 * @param dataFormat optional `data_format` attribute; TensorFlow's default (`"NHWC"`) is used when null.
 */
internal fun Ops.maxPoolWithStaticWindow(
    input: Operand<TFloat32>,
    poolSize: IntArray,
    strides: IntArray,
    padding: String,
    dataFormat: String? = null
): Operand<TFloat32> {
    val builder = scope().opBuilder(MAX_POOL_OP, MAX_POOL_OP)
    builder.addInput(input.asOutput())
    builder.setAttr("ksize", LongArray(poolSize.size) { poolSize[it].toLong() })
    builder.setAttr("strides", LongArray(strides.size) { strides[it].toLong() })
    builder.setAttr("padding", padding)
    if (dataFormat != null) builder.setAttr("data_format", dataFormat)
    return builder.build().output(0)
}
