/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import org.junit.jupiter.api.condition.EnabledIf
import org.tensorflow.DeviceSpec
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import kotlin.math.abs

/**
 * Checks that the ops KotlinDL's layers emit produce the same numbers on Metal as on the CPU.
 *
 * This covers the operations behind the common layer types rather than the layers themselves,
 * because `:tensorflow-metal` cannot depend on the layer test fixtures without a dependency cycle.
 * Ops with no Metal kernel are included on purpose: under soft placement they fall back to the CPU,
 * and that fallback must still yield correct results rather than silently diverge.
 *
 * Skipped unless a real plugin is available; see [MetalDeviceTest].
 */
@EnabledIf("pluginIsAvailable")
class MetalNumericsTest {

    /** Relative tolerance. Metal may reassociate floating-point work, so exact equality is not required. */
    private val tolerance = 1e-4f

    @TestFactory
    fun opsProduceTheSameResultsOnMetalAndCpu(): List<DynamicTest> {
        MetalAcceleration.enable()

        val cases = linkedMapOf<String, (Ops) -> Operand<TFloat32>>(
            "conv2d" to { tf ->
                tf.nn.conv2d(image(tf), kernel(tf), listOf(1L, 1L, 1L, 1L), "SAME")
            },
            "conv2d.valid" to { tf ->
                tf.nn.conv2d(image(tf), kernel(tf), listOf(1L, 2L, 2L, 1L), "VALID")
            },
            "depthwiseConv2d" to { tf ->
                tf.nn.depthwiseConv2dNative(image(tf), kernel(tf), listOf(1L, 1L, 1L, 1L), "SAME")
            },
            // The attribute form, which is what MaxPool2D emits since it is the one Metal implements.
            "maxPool" to { tf -> pool(tf, "MaxPool") },
            "avgPool" to { tf ->
                tf.nn.avgPool(image(tf), listOf(1L, 2L, 2L, 1L), listOf(1L, 2L, 2L, 1L), "SAME")
            },
            "matMul" to { tf -> tf.linalg.matMul(matrix(tf), matrix(tf)) },
            "biasAdd" to { tf -> tf.nn.biasAdd(matrix(tf), tf.constant(floatArrayOf(0.5f, -0.25f, 1.5f, 0f))) },
            "relu" to { tf -> tf.nn.relu(matrix(tf)) },
            "relu6" to { tf -> tf.nn.relu6(matrix(tf)) },
            "sigmoid" to { tf -> tf.math.sigmoid(matrix(tf)) },
            "tanh" to { tf -> tf.math.tanh(matrix(tf)) },
            "softmax" to { tf -> tf.nn.softmax(matrix(tf)) },
            "logSoftmax" to { tf -> tf.nn.logSoftmax(matrix(tf)) },
            // No Metal kernel: exercises the soft-placement fallback path.
            "elu (falls back)" to { tf -> tf.nn.elu(matrix(tf)) },
            "softplus (falls back)" to { tf -> tf.math.softplus(matrix(tf)) },
            "mean" to { tf -> tf.math.mean(matrix(tf), tf.constant(intArrayOf(0, 1))) },
            "sum" to { tf -> tf.reduceSum(matrix(tf), tf.constant(intArrayOf(0, 1))) },
            "squaredDifference" to { tf -> tf.math.squaredDifference(matrix(tf), tf.constant(0.25f)) }
        )

        return cases.map { (name, build) ->
            DynamicTest.dynamicTest(name) {
                val onCpu = run(build, DeviceSpec.DeviceType.CPU)
                val onGpu = run(build, DeviceSpec.DeviceType.GPU)

                assertEquals(onCpu.size, onGpu.size, "$name: result sizes differ")
                for (i in onCpu.indices) {
                    val scale = maxOf(1.0f, abs(onCpu[i]))
                    assertTrue(
                        abs(onCpu[i] - onGpu[i]) <= tolerance * scale,
                        "$name: element $i differs — CPU ${onCpu[i]} vs Metal ${onGpu[i]}"
                    )
                }
            }
        }
    }

    private fun run(build: (Ops) -> Operand<TFloat32>, device: DeviceSpec.DeviceType): FloatArray {
        Graph().use { graph ->
            val tf = Ops.create(graph)
                .withDevice(DeviceSpec.newBuilder().deviceType(device).build())
            val output = build(tf)
            Session(graph, metalGpuConfiguration().toTensorFlowSessionConfig()).use { session ->
                session.runner().fetch(output).run().use { result ->
                    val tensor = result.get(0) as TFloat32
                    val values = FloatArray(tensor.size().toInt())
                    tensor.asRawTensor().data().asFloats().read(values)
                    return values
                }
            }
        }
    }

    /** Builds the attribute-form pooling op, mirroring what MaxPool2D emits. */
    private fun pool(tf: Ops, op: String): Operand<TFloat32> {
        val builder = tf.scope().opBuilder(op, op)
        builder.addInput(image(tf).asOutput())
        builder.setAttr("ksize", longArrayOf(1, 2, 2, 1))
        builder.setAttr("strides", longArrayOf(1, 2, 2, 1))
        builder.setAttr("padding", "SAME")
        return builder.build().output(0)
    }

    // Deterministic inputs: a data pipeline would add noise to a comparison that is about kernels.
    private fun image(tf: Ops): Operand<TFloat32> {
        val values = FloatArray(1 * 8 * 8 * 4) { (it % 17) * 0.125f - 1.0f }
        return tf.reshape(tf.constant(values), tf.constant(longArrayOf(1, 8, 8, 4)))
    }

    private fun kernel(tf: Ops): Operand<TFloat32> {
        val values = FloatArray(3 * 3 * 4 * 4) { (it % 7) * 0.1f - 0.3f }
        return tf.reshape(tf.constant(values), tf.constant(longArrayOf(3, 3, 4, 4)))
    }

    private fun matrix(tf: Ops): Operand<TFloat32> {
        val values = FloatArray(4 * 4) { (it % 9) * 0.5f - 2.0f }
        return tf.reshape(tf.constant(values), tf.constant(longArrayOf(4, 4)))
    }

    companion object {
        @JvmStatic
        fun pluginIsAvailable(): Boolean =
            MetalAcceleration.isSupportedPlatform && MetalPlugin.locate() != null
    }
}
