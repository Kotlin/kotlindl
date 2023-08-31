project.setDescription("This module contains the Kotlin API implemetantion code for building, training, and evaluating the Deep Learning models.")

apply plugin: 'kotlin-multiplatform'
apply plugin: 'com.android.library'

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = '1.8'
        }
        testRuns["test"].executionTask.configure {
            useJUnitPlatform()
        }
    }
    android {
        publishLibraryVariants("release")
    }
    sourceSets {
        commonMain {
            dependencies {
                api project(':api')
            }
        }
        jvmMain {
            dependencies {
                api 'com.github.doyaaaaaken:kotlin-csv-jvm:0.7.3' // for csv parsing
                api 'io.github.microutils:kotlin-logging:2.1.21' // for logging
                api 'io.jhdf:jhdf:0.5.7' // for hdf5 parsing
                api 'com.beust:klaxon:5.5'
                implementation 'com.twelvemonkeys.imageio:imageio-jpeg:3.8.2'
            }
        }
        jvmTest {
            dependencies {
                implementation 'org.junit.jupiter:junit-jupiter-api:5.8.2'
                implementation 'org.junit.jupiter:junit-jupiter-engine:5.8.2'
                implementation 'org.junit.jupiter:junit-jupiter-params:5.8.2'
            }
        }
        androidMain {
            dependencies {
                api 'androidx.camera:camera-core:1.0.0-rc03'
            }
        }
    }
    explicitApiWarning()
}

android {
    compileSdkVersion 31
    namespace = 'org.jetbrains.kotlinx.dl.impl'
    defaultConfig {
        minSdkVersion 24
        targetSdkVersion 31
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}