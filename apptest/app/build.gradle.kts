plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
    id("com.chaquo.python")
}

android {
    namespace = "com.example.myapplication"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.myapplication"
        minSdk = 35
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("x86", "x86_64", "arm64-v8a") // "x86", "x86_64", "arm64-v8a"
        }
    }

    androidResources {
        noCompress += listOf("onnx", "pkl")
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        viewBinding = true
    }
}

chaquopy {
    defaultConfig {
        // buildPython("/usr/local/bin/python3.10")
        buildPython("/Users/seul/.pyenv/shims/python3")
        // buildPython("C:\\Users\\SSAFY\\AppData\\Local\\Programs\\Python\\Python38\\python.exe")
        // buildPython("C:\\Users\\SSAFY\\anaconda3\\python.exe")
        pip {
            install("librosa==0.8.1")
            install("resampy==0.2.2")
            install("numba==0.48.0")

            install("tensorflow==2.1.0")
            install("protobuf==3.20.3")

            install("matplotlib")
            install("numpy")
            install("pillow")
            install("scipy")

            // scikit-learn 1.3.2
            install("scikit-learn")
            install("pandas")
            install("tqdm")
        }
    }
}



dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.navigation:navigation-fragment-ktx:2.8.9")
    implementation("androidx.navigation:navigation-ui-ktx:2.8.9")

    implementation("com.google.ai.edge.litert:litert:1.2.0")
    implementation("com.github.wendykierp:JTransforms:3.1")
//    implementation("org.tensorflow:tensorflow-lite:2.18.0")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.15.1")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar"))))

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
