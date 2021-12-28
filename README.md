# Edge Detection

This is an application made to detect edges on an image with a simple drag and drop user interface. The algorithm can run on a CPU or on an NVIDIA GPU in order to shorten the process.

![alt text](https://i.postimg.cc/KcTp1Srh/edge-detector.jpg)

## Requirements
- .net core 5
- CUDA toolkit (only for GPU version)

## Interface
The application consists of two main panels - Input and Output. In the input part you can either drag a drop an image or browse the file explorer.

The output panel is where your generated image with detected edges will appear. In order to do so you will first need to set your desired options or leave them with their default values.

Once you are ready you can generate the image with the *Execute* button or check the *Enable auto execute* checkbox so that the algorithm automatically runs with any option changes.

When the algorithm is running you should see a *Processing* indicator in the bottom left corner. It will then change to *Ready* when the image is succesfully generated and shown in the Output panel.

## Options
There are many options which allow you to change how to algorithm will be executed.

**GPU/CPU** radio buttons allow you to either run the algorithm on your processor or your graphics card provided that you have a NVIDIA graphics card. If you try to run it without a proper hardware you should see a black empty image in the output panel in which case you should change the method to CPU.

**Threshold** slider allows you to pick how sensitive the detection method will be to different pixel values - i.e. a pixel will appear as white only if it is above the threshold values.

**Sigma** regulates gaussian filter. Lower value - less blur.

**Modify mask** opens a panel in which you can change the desired edge mask. You can use it with the default values or you can modify them - for example if you only want to detect horizontal edges you can set the *Y* mask values to 0.

**Gausian Filter** enables Gaussian Filter - blurs image (GPU Only).

**Non maximum suppression** enables Non Maximum Suppression - suppressing all pixels that are not part in local maxima (GPU Only).
