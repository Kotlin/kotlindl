pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
        mavenCentral()
    }
    resolutionStrategy {
        eachPlugin {
            if (requested.id.namespace == 'com.android' || requested.id.name == 'kotlin-android-extensions') {
                useModule('com.android.tools.build:gradle:7.2.0')
            }
        }
    }
}

rootProject.name = 'KotlinDL'
include("api")
include("impl")
include("tensorflow")
include("visualization")
include("examples")
include("dataset")
include("onnx")
include("gradlePlugin")