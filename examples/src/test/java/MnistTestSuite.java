/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */


import org.jetbrains.kotlinx.dl.logging.api.LogFactoryKt;
import org.jetbrains.kotlinx.dl.logging.core.DefaultLogFactory;
import org.junit.jupiter.api.Test;

public class MnistTestSuite {
    public static final String[] EMPTY_ARGS = new String[0];

    @Test
    public void lenetClassicTest() {
        LogFactoryKt.setGlobalLogFactory(DefaultLogFactory.INSTANCE);
        DefaultLogFactory.INSTANCE.setup(DefaultLogFactory.INSTANCE.getDefaultConfig());
        LeNetClassic.main(EMPTY_ARGS);
    }
}
