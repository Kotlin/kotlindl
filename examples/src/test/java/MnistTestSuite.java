/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */


import org.junit.jupiter.api.Test;

public class MnistTestSuite {
    public static final String[] EMPTY_ARGS = new String[0];

    @Test
    public void lenetClassicTest() {
        LeNetClassic.main(EMPTY_ARGS);
    }
}
