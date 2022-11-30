/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import ai.onnxruntime.*

/**
 * Convenience functions for processing [OrtSession.Result].
 */
public object OrtSessionResultConversions {
    /**
     * Returns the output at [index] as a [FloatArray] with its shape.
     */
    public fun OrtSession.Result.getFloatArrayWithShape(index: Int): Pair<FloatArray, LongArray> {
        return get(index).getFloatArrayWithShape()
    }

    /**
     * Returns the output at [index] as a [FloatArray].
     */
    public fun OrtSession.Result.getFloatArray(index: Int): FloatArray {
        return getFloatArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as a [FloatArray] with its shape.
     */
    public fun OrtSession.Result.getFloatArrayWithShape(name: String): Pair<FloatArray, LongArray> {
        return get(name).get().getFloatArrayWithShape()
    }

    /**
     * Returns the output by [name] as a [FloatArray].
     */
    public fun OrtSession.Result.getFloatArray(name: String): FloatArray {
        return getFloatArrayWithShape(name).first
    }

    private fun OnnxValue.getFloatArrayWithShape(): Pair<FloatArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getFloatArray", OnnxJavaType.FLOAT)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).floatBuffer.array() to shape
    }

    /**
     * Returns the output at [index] as a [DoubleArray] with its shape.
     */
    public fun OrtSession.Result.getDoubleArrayWithShape(index: Int): Pair<DoubleArray, LongArray> {
        return get(index).getDoubleArrayWithShape()
    }

    /**
     * Returns the output at [index] as a [DoubleArray].
     */
    public fun OrtSession.Result.getDoubleArray(index: Int): DoubleArray {
        return getDoubleArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as a [DoubleArray] with its shape.
     */
    public fun OrtSession.Result.getDoubleArrayWithShape(name: String): Pair<DoubleArray, LongArray> {
        return get(name).get().getDoubleArrayWithShape()
    }

    /**
     * Returns the output by [name] as a [DoubleArray].
     */
    public fun OrtSession.Result.getDoubleArray(name: String): DoubleArray {
        return getDoubleArrayWithShape(name).first
    }

    private fun OnnxValue.getDoubleArrayWithShape(): Pair<DoubleArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getDoubleArray", OnnxJavaType.DOUBLE)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).doubleBuffer.array() to shape
    }

    /**
     * Returns the output at [index] as a [LongArray] with its shape.
     */
    public fun OrtSession.Result.getLongArrayWithShape(index: Int): Pair<LongArray, LongArray> {
        return get(index).getLongArrayWithShape()
    }

    /**
     * Returns the output at [index] as a [LongArray].
     */
    public fun OrtSession.Result.getLongArray(index: Int): LongArray {
        return getLongArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as a [LongArray] with its shape.
     */
    public fun OrtSession.Result.getLongArrayWithShape(name: String): Pair<LongArray, LongArray> {
        return get(name).get().getLongArrayWithShape()
    }

    /**
     * Returns the output by [name] as a [FloatArray].
     */
    public fun OrtSession.Result.getLongArray(name: String): LongArray {
        return getLongArrayWithShape(name).first
    }

    private fun OnnxValue.getLongArrayWithShape(): Pair<LongArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getLongArray", OnnxJavaType.INT64)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).longBuffer.array() to shape
    }

    /**
     * Returns the output at [index] as an [IntArray] with its shape.
     */
    public fun OrtSession.Result.getIntArrayWithShape(index: Int): Pair<IntArray, LongArray> {
        return get(index).getIntArrayWithShape()
    }

    /**
     * Returns the output at [index] as an [IntArray].
     */
    public fun OrtSession.Result.getIntArray(index: Int): IntArray {
        return getIntArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as an [IntArray] with its shape.
     */
    public fun OrtSession.Result.getIntArrayWithShape(name: String): Pair<IntArray, LongArray> {
        return get(name).get().getIntArrayWithShape()
    }

    /**
     * Returns the output by [name] as an [IntArray].
     */
    public fun OrtSession.Result.getIntArray(name: String): IntArray {
        return getIntArrayWithShape(name).first
    }

    private fun OnnxValue.getIntArrayWithShape(): Pair<IntArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getIntArray", OnnxJavaType.INT32)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).intBuffer.array() to shape
    }

    /**
     * Returns the output at [index] as a [ShortArray] with its shape.
     */
    public fun OrtSession.Result.getShortArrayWithShape(index: Int): Pair<ShortArray, LongArray> {
        return get(index).getShortArrayWithShape()
    }

    /**
     * Returns the output at [index] as a [ShortArray].
     */
    public fun OrtSession.Result.getShortArray(index: Int): ShortArray {
        return getShortArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as a [ShortArray] with its shape.
     */
    public fun OrtSession.Result.getShortArrayWithShape(name: String): Pair<ShortArray, LongArray> {
        return get(name).get().getShortArrayWithShape()
    }

    /**
     * Returns the output by [name] as a [ShortArray].
     */
    public fun OrtSession.Result.getShortArray(name: String): ShortArray {
        return getShortArrayWithShape(name).first
    }

    private fun OnnxValue.getShortArrayWithShape(): Pair<ShortArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getShortArray", OnnxJavaType.INT16)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).shortBuffer.array() to shape
    }

    /**
     * Returns the output at [index] as a [ByteArray] with its shape.
     */
    public fun OrtSession.Result.getByteArrayWithShape(index: Int): Pair<ByteArray, LongArray> {
        return get(index).getByteArrayWithShape()
    }

    /**
     * Returns the output at [index] as a [ByteArray].
     */
    public fun OrtSession.Result.getByteArray(index: Int): ByteArray {
        return getByteArrayWithShape(index).first
    }

    /**
     * Returns the output by [name] as a [ByteArray] with its shape.
     */
    public fun OrtSession.Result.getByteArrayWithShape(name: String): Pair<ByteArray, LongArray> {
        return get(name).get().getByteArrayWithShape()
    }

    /**
     * Returns the output by [name] as a [ByteArray].
     */
    public fun OrtSession.Result.getByteArray(name: String): ByteArray {
        return getByteArrayWithShape(name).first
    }

    private fun OnnxValue.getByteArrayWithShape(): Pair<ByteArray, LongArray> {
        throwIfOutputNotSupported(info, toString(), "getByteArray", OnnxJavaType.STRING)
        val shape = (info as TensorInfo).shape
        return (this as OnnxTensor).byteBuffer.array() to shape
    }

    /**
     * Returns the output by [name] as an Array<FloatArray>. This operation could be slow for high dimensional tensors,
     * in which case [getFloatArray] should be used.
     */
    public fun OrtSession.Result.get2DFloatArray(name: String): Array<FloatArray> {
        return get(name).get().get2DFloatArray()
    }

    /**
     * Returns the output at [index] as an Array<FloatArray>. This operation could be slow for high dimensional tensors,
     * in which case [getFloatArray] should be used.
     */
    public fun OrtSession.Result.get2DFloatArray(index: Int): Array<FloatArray> {
        return get(index).get2DFloatArray()
    }

    @Suppress("UNCHECKED_CAST")
    private fun OnnxValue.get2DFloatArray(): Array<FloatArray> {
        throwIfOutputNotSupported(info, toString(), "get2DFloatArray", OnnxJavaType.FLOAT)
        val shape = (info as TensorInfo).shape
        val depth = shape.size - 2
        require(depth >= 0 && shape.slice(0 until depth).all { it == 1L }) {
            "Output of shape $shape can't be converted to the Array<FloatArray>."
        }
        var result = value as Array<*>
        repeat(depth) {
            result = result[0] as Array<*>
        }
        return result as Array<FloatArray>
    }

    /**
     * Returns all values from this [OrtSession.Result]. This operation could be slow for high dimensional tensors,
     * in which case functions that return one dimensional array such as [getFloatArray] or [getLongArray] should be used.
     * @see OnnxValue.getValue
     */
    public fun OrtSession.Result.getValues(): Map<String, Any> = associate { it.key to it.value.value }

    /**
     * Checks if [valueInfo] corresponds to a Tensor of the specified [type].
     * If it does not satisfy the requirements, exception with a message containing [valueName] and calling [method] name is thrown.
     */
    internal fun throwIfOutputNotSupported(
        valueInfo: ValueInfo,
        valueName: String,
        method: String,
        type: OnnxJavaType? = null
    ) {
        val typeString = type?.toString()?.let { "$it " } ?: ""
        require(valueInfo !is MapInfo) { "Output '$valueName' is a Map, but currently method '$method' supports only ${typeString}Tensor outputs." }
        require(valueInfo !is SequenceInfo) { "Output '$valueName' is a Sequence, but currently method '$method' supports ${typeString}Tensor outputs." }
        if (type != null) {
            require(valueInfo is TensorInfo && valueInfo.type == type) { "Currently method '$method' supports only ${typeString}Tensor outputs, but output '$valueName' is not a ${typeString}Tensor." }
        }
    }
}