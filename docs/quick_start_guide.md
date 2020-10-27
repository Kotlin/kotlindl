# Quick Start Guide

Whether you want to add Kotlin DL into your project, or experiment with it in a Jupyter Notebook, first you'll need to 
set up appropriate dependencies.  

## Working with Kotlin DL in an IDE
1. Open a Kotlin project where you want to use Kotlin DL, or create a new Kotlin project in IntelliJ IDEA as described 
in [Kotlin documentation](https://kotlinlang.org/docs/tutorials/jvm-get-started.html).
2. Add the Kotlin DL dependency to your project's build file. 
    * If you're using Gradle as build system, add the following to the `build.gradle.kts` file:
    ```kotlin
    repositories {
        maven(url = "https://kotlin.bintray.com/kotlin-datascience")
    }
    
    dependencies {
       implementation ("org.jetbrains.kotlin.kotlin-dl:api:[Kotlin DL version]]")
    }
    ```  
   * If your project is built with Maven, add Kotlin DL to your `pom.xml`: 
   ```xml
   <repositories>
       <repository>
           <id>jcenter</id>
           <name>jcenter</name>
           <url>https://kotlin.bintray.com/kotlin-datascience</url>
       </repository>
   </repositories>
   
   <dependency>
     <groupId>org.jetbrains.kotlin.kotlin-dl</groupId>
     <artifactId>api</artifactId>
     <version>[Kotlin DL version]</version>
     <type>pom</type>
   </dependency>
   ```
 
You can also take advantage of Kotlin DL functionality in any existing Java project, even if you have no other Kotlin 
code in it yet. Check out [these instructions](https://kotlinlang.org/docs/tutorials/mixing-java-kotlin-intellij.html#adding-kotlin-source-code-to-an-existing-java-project) 
on how to add Kotlin code  in your existing Java code base. 
  
That's it! Now you're ready to [build your first neural network](create_your_first_nn.md). 

## Working with Kotlin DL in a Jupyter Notebook
If you want to experiment with Kotlin DL, and use it interactively, you can choose to work with it in 
a Jupyter Notebook. 

1. To set up Jupyter Notebook itself, first you need to have [Python](https://www.python.org/) installed on your machine. 
2. Then, the easiest way to get started is to install [Anaconda distribution](https://www.anaconda.com/products/individual) that includes Jupyter Notebook. 
3. For the Kotlin kernel to work, make sure you have Java v.8 or later installed. 
4. Once you do, add the Kotlin Kernel to Jupyter Notebook with the following command: 

    ```conda install -c jetbrains kotlin-jupyter-kernel```
5. Start your Jupyter Notebook with `jupyter notebook`
6. Once Jupyter Notebook starts and opens in your browser, you can create a new Kotlin notebook from the UI. 
7. In your new Kotlin notebook, add a dependency for Kotlin DL:
    ```
   @file:Repository("https://kotlin.bintray.com/kotlin-datascience")
   @file:DependsOn("org.jetbrains.kotlin.kotlin-dl:api:[Kotlin DL version]")
   ```

All set! Next, you can start [building your first neural network](create_your_first_nn.md).
 
