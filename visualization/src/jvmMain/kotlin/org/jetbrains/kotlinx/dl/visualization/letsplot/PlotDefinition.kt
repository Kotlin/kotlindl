/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.letsplot

import org.jetbrains.letsPlot.geom.geomRaster
import org.jetbrains.letsPlot.intern.Scale
import org.jetbrains.letsPlot.scale.scaleFillGrey
import org.jetbrains.letsPlot.scale.scaleFillHue
import org.jetbrains.letsPlot.themes.theme

/**
 * PlotFeature represents the filling options for plots that defines its color scale.
 * Used mainly with predefined scales that are available as properties of companion
 * object but can be parametrized with some custom [Scale] for filling from lets-plot
 * library.
 *
 * @property scale defining the feature of color scale for plot
 * @constructor Create [PlotFeature] with defined scale
 */
class PlotFeature(val scale: Scale) {

    companion object {
        val GRAY: PlotFeature = PlotFeature(scaleFillGrey())
        val HUE: PlotFeature = PlotFeature(scaleFillHue())
    }
}

internal val FEATURE_MAP_THEME =
    geomRaster(showLegend = false) +
            theme()
//.axisTitleBlank()
//.axisTextBlank()
//.axisTicksBlank()
