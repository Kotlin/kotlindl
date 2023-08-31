project.description = "This module contains the Kotlin API for building, training, and evaluating TensorFlow models."

plugins {
    kotlin("jvm")
}

dependencies {
    api(project(":dataset"))
    api(project(":api"))
    api(project(":impl"))
    api(group = "org.tensorflow", name = "tensorflow", version = "1.15.0")
    api("com.beust:klaxon:5.5")
    api("io.github.microutils:kotlin-logging:2.1.21")
    testImplementation("ch.qos.logback:logback-classic:1.2.11")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.8.2")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.8.2")
}

tasks.compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.compileTestKotlin {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.named<Test>("test") {
    useJUnitPlatform()
}

kotlin {
    explicitApiWarning()
}

tasks.register<Jar>("sourcesJar") {
    classifier = "sources"
    from(sourceSets["main"].allSource)
}

artifacts {
    archives(tasks.named("sourcesJar"))
}