# Quick Start Guide

Whether you want to add Kotlin DL to your project or experiment with it in a Jupyter Notebook, 
first you'll need to set up the appropriate dependencies.  

## Working with Kotlin DL in an IDE
1. Open a Kotlin project where you want to use Kotlin DL, or create a new Kotlin project in IntelliJ IDEA as described in the [Kotlin documentation](https://kotlinlang.org/docs/tutorials/jvm-get-started.html).
2. Add the Kotlin DL dependency to your project's build file. 
    * If you're using Gradle as the build system, add the following to the `build.gradle` file:
    ```groovy
   repositories {
       mavenCentral()
   }
   
   dependencies {
       implementation 'org.jetbrains.kotlinx:kotlin-deeplearning-api:[KOTLIN-DL-VERSION]'
   }
    ```  
   * For `build.gradle.kts`: 
   ```kotlin
   repositories {
       mavenCentral()
   }
       
   dependencies {
       implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-api:[KOTLIN-DL-VERSION]")
   }
   ```   
   * If your project is built with Maven, add Kotlin DL to your `pom.xml`: 
   ```xml
   <dependency>
     <groupId>org.jetbrains.kotlinx</groupId>
     <artifactId>kotlin-deeplearning-api</artifactId>
     <version>[KOTLIN-DL-VERSION]</version>
     <type>pom</type>
   </dependency>
   ```
 
You can use the Kotlin DL functionality in any existing Java project, even if you don’t have any other Kotlin code in it yet. 
Check out [these instructions](https://kotlinlang.org/docs/tutorials/mixing-java-kotlin-intellij.html#adding-kotlin-source-code-to-an-existing-java-project) 
on how to add Kotlin code to your existing Java codebase. 
  
That's it! Now you're ready to [build your first neural network](create_your_first_nn.md). 

## Working with Kotlin DL in Android Studio
1. Open an Android project where you want to use Kotlin DL, or create a new Android project in Android Studio as described in the [Android documentation](https://developer.android.com/training/basics/firstapp).
2. Add the Kotlin DL dependency to your project's build files. 
    * Add the following to the top-level `build.gradle` file:
   ```kotlin
   repositories {
       mavenCentral()
   }
   ```
   * Add the following to the `build.gradle` file in the application module:
   ```kotlin
   dependencies {
       implementation 'org.jetbrains.kotlinx:kotlin-deeplearning-onnx:[KOTLIN-DL-VERSION]'
   }
   ```
3. Please check out the [Documentation](https://kotlin.github.io/kotlindl/) and [Sample Android App](https://github.com/Kotlin/kotlindl-app-sample) for more details.

## Working with Kotlin DL in a Jupyter Notebook
If you want to experiment with Kotlin DL, and use it interactively, you can choose to work with it in a [Jupyter Notebook](https://jupyter.org). 
In this case, you will need to install Jupyter, add the 
[Kotlin kernel](https://github.com/Kotlin/kotlin-jupyter), and set up the Kotlin DL dependency in the notebook. 
Here are step-by-step instructions to help you get started:

1. To set up Jupyter Notebook, you need to first have [Python](https://www.python.org/) installed on your machine. 
2. The next step is to install the [Anaconda distribution](https://www.anaconda.com/products/individual) that includes Jupyter Notebook. 
3. For the Kotlin kernel to work, make sure you have Java v.8 or later installed. 
4. Once installed, add the Kotlin Kernel to Jupyter Notebook with the following command: 

    ```conda install -c jetbrains kotlin-jupyter-kernel```
5. Start your Jupyter Notebook from the command line using `jupyter notebook`
6. Once Jupyter Notebook is open in your browser, you can create a new Kotlin notebook from the UI. 
7. In your new Kotlin notebook, add a dependency for Kotlin DL:
```
   @file:DependsOn("org.jetbrains.kotlinx:kotlin-deeplearning-api:[KOTLIN-DL-VERSION]")
```

You are now all set! Next, you can start [building your first neural network](create_your_first_nn.md).
 
