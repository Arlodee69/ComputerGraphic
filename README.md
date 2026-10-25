# Build and Run

Requires CMake, Git, a C++20 compiler, and Vulkan SDK with `glslc` in PATH.

## Lab 1

```powershell
cd D:\CG\vulkan-starter-app
cmake --preset mingw-debug
cmake --build build-debug --parallel
.\build-debug\vulkan-starter-app.exe
```

## Lab 2

```powershell
cd D:\CG\vulkan-starter-app-lab2
cmake --preset mingw-debug
cmake --build build-debug --parallel
.\build-debug\vulkan-starter-app.exe
```

For Visual Studio 2022, use `msvc-debug` and run
`.\build-debug\Debug\vulkan-starter-app.exe` from the project root.
Close the application before rebuilding. After source or shader changes, repeat
the build and run commands. Reconfigure after changing CMake settings.
