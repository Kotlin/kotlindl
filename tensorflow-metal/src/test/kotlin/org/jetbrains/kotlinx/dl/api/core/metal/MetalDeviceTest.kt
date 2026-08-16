/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestMethodOrder
import org.junit.jupiter.api.MethodOrderer
import org.junit.jupiter.api.Order
import org.junit.jupiter.api.condition.EnabledIf
import org.tensorflow.DeviceSpec
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

/**
 * End-to-end tests against a real Metal device.
 *
 * Skipped unless `libmetal_plugin.dylib` can actually be found, because KotlinDL does not bundle
 * Apple's binary. To run them, point the build at a copy:
 *
 * ```
 * ./gradlew :tensorflow-metal:test -Dkotlindl.metal.plugin=/path/to/libmetal_plugin.dylib
 * ```
 *
 * See [MetalPlugin] for the other locations that are searched.
 */
@EnabledIf("pluginIsAvailable")
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
class MetalDeviceTest {

    @Test
    @Order(1)
    fun enableRegistersAUsableMetalDevice() {
        when (val status = MetalAcceleration.enable()) {
            is MetalStatus.Enabled -> assertTrue(MetalAcceleration.isEnabled)
            is MetalStatus.Unavailable -> throw AssertionError("Metal should be available here: ${status.reason}")
        }
    }

    @Test
    @Order(2)
    fun enableIsIdempotent() {
        val first = MetalAcceleration.enable()
        val second = MetalAcceleration.enable()
        assertEquals(first, second, "Repeated calls must return the memoised result.")
    }

    @Test
    @Order(3)
    fun matrixMultiplyRunsOnTheGpuAndMatchesTheCpu() {
        MetalAcceleration.enable()

        val a = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f))
        val b = arrayOf(floatArrayOf(5f, 6f), floatArrayOf(7f, 8f))

        val onGpu = matMul(a, b, DeviceSpec.DeviceType.GPU)
        val onCpu = matMul(a, b, DeviceSpec.DeviceType.CPU)

        assertEquals(onCpu.toList(), onGpu.toList(), "Metal and CPU results must agree.")
        assertEquals(listOf(19f, 22f, 43f, 50f), onGpu.toList())
    }

    /**
     * The behaviour that dictates [metalGpuConfiguration]'s default: the plugin has no Metal kernel
     * for `Assign`, so a graph whose variables are pinned hard to the GPU cannot be placed. Soft
     * placement is what makes training work.
     */
    @Test
    @Order(4)
    fun trainingRequiresSoftPlacement() {
        MetalAcceleration.enable()

        val withoutSoftPlacement = runCatching { trainOneStep(allowSoftPlacement = false) }
        assertTrue(
            withoutSoftPlacement.isFailure,
            "Expected hard GPU placement to fail because Assign has no Metal kernel."
        )
        assertTrue(
            withoutSoftPlacement.exceptionOrNull()?.message.orEmpty().contains("device", ignoreCase = true),
            "Expected a placement failure, got: ${withoutSoftPlacement.exceptionOrNull()?.message}"
        )

        val (before, after) = trainOneStep(allowSoftPlacement = true)
        assertTrue(after < before, "Loss should decrease after a gradient step: $before -> $after")
    }

    private fun matMul(
        a: Array<FloatArray>,
        b: Array<FloatArray>,
        device: DeviceSpec.DeviceType
    ): FloatArray {
        Graph().use { graph ->
            val tf = Ops.create(graph)
                .withDevice(DeviceSpec.newBuilder().deviceType(device).build())
            val product = tf.linalg.matMul(tf.constant(a), tf.constant(b))
            Session(graph, metalGpuConfiguration().toTensorFlowSessionConfig()).use { session ->
                session.runner().fetch(product).run().use { result ->
                    val tensor = result.get(0) as TFloat32
                    return floatArrayOf(
                        tensor.getFloat(0, 0), tensor.getFloat(0, 1),
                        tensor.getFloat(1, 0), tensor.getFloat(1, 1)
                    )
                }
            }
        }
    }

    /** Runs a single gradient-descent step pinned to the GPU; returns loss before and after. */
    private fun trainOneStep(allowSoftPlacement: Boolean): Pair<Float, Float> {
        Graph().use { graph ->
            val tf = Ops.create(graph)
                .withDevice(DeviceSpec.newBuilder().deviceType(DeviceSpec.DeviceType.GPU).build())

            val weights = tf.variable(tf.constant(arrayOf(floatArrayOf(1f, 1f), floatArrayOf(1f, 1f))))
            val x = tf.constant(arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f)))
            val y = tf.constant(arrayOf(floatArrayOf(1f, 0f), floatArrayOf(0f, 1f)))
            val difference = tf.math.sub(tf.linalg.matMul(x, weights), y)
            val loss = tf.math.mean(tf.math.square(difference), tf.constant(intArrayOf(0, 1)))

            @Suppress("UNCHECKED_CAST")
            val gradient = tf.gradients(loss, listOf(weights)).dy<TFloat32>(0)
            val update = tf.train.applyGradientDescent(weights, tf.constant(0.1f), gradient)

            val config = metalGpuConfiguration(allowSoftPlacement = allowSoftPlacement)
            Session(graph, config.toTensorFlowSessionConfig()).use { session ->
                session.initialize()
                val before = fetchLoss(session, loss)
                session.runner().addTarget(update.op()).run().close()
                return before to fetchLoss(session, loss)
            }
        }
    }

    private fun fetchLoss(session: Session, loss: org.tensorflow.Operand<TFloat32>): Float =
        session.runner().fetch(loss).run().use { (it.get(0) as TFloat32).getFloat() }

    companion object {
        @JvmStatic
        fun pluginIsAvailable(): Boolean =
            MetalAcceleration.isSupportedPlatform && MetalPlugin.locate() != null
    }
}
