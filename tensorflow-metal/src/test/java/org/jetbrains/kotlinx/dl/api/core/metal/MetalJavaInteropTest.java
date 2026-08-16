/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Locks in how this API looks from Java. KotlinDL is documented as usable from Java projects, and
 * without {@code @JvmStatic} a Kotlin {@code object} forces callers to write
 * {@code MetalAcceleration.INSTANCE.enable(...)}, which is not an API anyone would design on purpose.
 *
 * This test is written in Java precisely so it fails to compile if those annotations are dropped.
 */
class MetalJavaInteropTest {

    @Test
    void entryPointsAreCallableStaticallyFromJava() {
        // Each of these is a static call. Removing @JvmStatic breaks compilation, not just style.
        boolean supported = MetalAcceleration.isSupportedPlatform();
        boolean enabled = MetalAcceleration.isEnabled();
        String message = MetalPlugin.notFoundMessage();

        assertNotNull(message);
        // Nothing is asserted about `supported` beyond it being reachable; its value depends on the
        // host, and MetalAccelerationTest already covers the platform logic.
        assertTrue(supported || !supported);
        assertFalse(enabled && !MetalAcceleration.isEnabled());
    }

    @Test
    void enableIsCallableFromJavaAndReportsRatherThanThrows() {
        MetalStatus status = MetalAcceleration.enable(Path.of("/definitely/not/a/real/libmetal_plugin.dylib"));
        assertNotNull(status);

        // The sealed hierarchy should be usable from Java too.
        if (status instanceof MetalStatus.Unavailable unavailable) {
            assertNotNull(unavailable.getReason());
        } else {
            assertNotNull(((MetalStatus.Enabled) status).getPluginPath());
        }
    }

    /** The no-argument overload must exist for Java, which has no default parameter values. */
    @Test
    void enableHasANoArgumentOverloadForJava() throws Exception {
        assertNotNull(MetalAcceleration.class.getMethod("enable"));
        assertNotNull(MetalAcceleration.class.getMethod("enable", Path.class));
    }
}
