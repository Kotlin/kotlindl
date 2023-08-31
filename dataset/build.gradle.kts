project.description = "A KotlinDL library provides Dataset API for better Kotlin programming for Deep Learning."

plugins {
    kotlin("multiplatform")
}

kotlin {
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "1.8"
        }
    }
    sourceSets {
        val jvmMain by getting {
            dependencies {
                api(project(":api"))
                api(project(":impl"))
            }
        }
    }
    explicitApiWarning()
}