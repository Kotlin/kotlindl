/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import jetbrains.letsPlot.geom.geomRaster
import jetbrains.letsPlot.intern.Scale
import jetbrains.letsPlot.scale.scaleFillGrey
import jetbrains.letsPlot.scale.scaleFillHue
import jetbrains.letsPlot.theme

class PlotFill(val scale: Scale) {
    companion object {
        val GRAY = PlotFill(scaleFillGrey())
        val HUE = PlotFill(scaleFillHue())
    }
}

internal val FEATURE_MAP_THEME =
    geomRaster(showLegend = false) +
            theme().axisTitleBlank()
                .axisTextBlank()
                .axisTicksBlank()
