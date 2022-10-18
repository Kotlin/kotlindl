/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.visualization.swing

import java.awt.Dimension
import javax.swing.JFrame
import javax.swing.JPanel

/**
 * Shows a frame in the center of the screen with the provided [title] and [content].
 */
fun showFrame(title: String, content: JPanel, size: Dimension? = null) {
    val frame = JFrame(title)
    frame.contentPane.add(content)
    if (size != null) {
        frame.size = size
    } else {
        frame.pack()
    }
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}