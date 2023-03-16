# IDAO-2021 qualifying round Kirill Rybkin
## Task
The task is to classify images of the impact of high-energy particles on the detector. Classes: 2 kinds of particles, and each of them can have one of 6 energy values.

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/0.jpg" width="400" height="400">
  
  2 types of particles (ER and NR), for each 6 energy values.
</div>

For training, only 3 types of energies are given for each class, the public score is calculated from them during the competition, and the private score is calculated from 3 classes of energies that are not in the training sample. This was done to check the quality of the model.

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/1.png" width="400" height="400">
  
  Separation of particles into training (L) and test (T) samples.
</div>


## What is the peculiarity of the data, or why convolutional networks will not help solve the problem?
First of all, it is worth remembering that we are working with projections of 3-dimensional results of interaction reactions of a high-energy particle onto a plane. This is stated in the video lecture on the competition, and there is information about this in the description of the competition. We need to classify images into 2 classes of particle types and 3 classes of energies.

Without going into the details of the imaging process, the spot is formed from several thousand dots / photons, the position of which is determined by a normal random distribution, and the values of the ‘brightness’ by a geometric distribution (and the sum of their brightnesses is equal to the energy of the original particle). These points are added pixel by pixel and form a spot. (It may seem that images with ER 10, 20 and 30 are of a different nature, but most likely there is just a smudge on the track).

It can be assumed that the very process of interaction of a particle with matter is mostly random, and hence the result of the interaction in the form of an image will retain this randomness. The shape of the spot (its radius) and the rate of fall of the 'brightness' from the center of the spot to the edges are not random and depend on the type of particle, the brightness depends on the type of particle energy, but the distribution of positions of points in the spot and their energy is absolutely random.

And it may seem that everything is extremely simple, and to determine the type of particle, it is enough to calculate its radius, and for energy, it is simply to sum the pixels of the spot, subtracting noise from them. And yes, by applying primitive filtering and adding the brightness of the pixels in the core, it is really possible to determine the energy of the particle very accurately (classification accuracy is 100%). However, there is a problem with determining the particle type. It is very easy to isolate particles with a long track or even just with an offset from the center, and since only ER 10, 20, 30 have an offset, determining their class is not a problem. But distinguishing particles ER 1 and 3 from NR 1, 3, 6 causes problems (accuracy by roc_auc_score=0.9993). The reason is precisely that, having such a small total energy, noises begin to introduce a serious error in determining the parameters of the spot.


## How is the definition of the spot parameters for the subsequent classification?
First of all, it should be clarified that we are talking about the classification of exactly round spots, since tracks can be classified without problems even without determining their characteristics. So, we have two types of spots, and in the image without noise from below, you can easily see that they differ slightly in size and, more importantly, in the rate of decrease in brightness from the center of the spot to the edges:

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/2.jpg" width="600" height="300">
  
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/3.jpg" width="600" height="300">
  
  Particle spots without noise from the competition video lecture (https://youtu.be/VzH_58yYz5k?t=2060)
</div>


It is these characteristics that I am trying to extract using the get_height_map function. In it, the image is cut into 30 fragments, in the form of circles of different radii, and the average value is calculated for each of them. This way I get something like a topographic height map, but since we know that the spot is a circle, instead of a 2d map, we have a graph of the change in the average height for each of the radius values.

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/4.jpg" width="500" height="400">
  
  Height map
</div>

### An example of image masks over which the height is averaged:
<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/5.png" width="900" height="300">
</div>

### The results of the get_height_map function for ER3 and NR6 particles:
As you can see, after smoothing the image, to minimize noise and normalize to the maximum, the particles give quite different plots of height change from the center to the edge of the spot.
<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/6.png" width="400" height="400">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/7.png" width="400" height="400">
</div>

## Final result

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/9.jpg" width="800" height="800">
</div>

As a final model, I used catboost to classify images by particle type and thresholds for classifying by energy. The final score of the model is about 0.999.

<div align="center">
  <img src="https://github.com/Oxonomy/IDAO-2021-Kirill/blob/main/8.png" width="500" height="500">
</div>

Since the data on public and private data were very different, I did not add public data predictions to the model, and this was not the best solution.

I mixed up the particle classes and the model predicted the particle type in reverse. Accordingly, auk was about 0.036.

However, the energy classification was correct and the mean absolute error was about 0.07. So the final result was not bad, the result in the LB was -38 and seventh place. Without this error, the result would have been around 900 and second place.
