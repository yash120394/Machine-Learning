# Stereo Matching using GMM and Gibbs Sampling

- im0.ppm (left) and im8.ppm (right) are the pictures taken by two different camera positions. 
- Let's call them XL and XR. For the (i,j)-th pixel in the right image, XR (i;j;:), which is a 3-d vector of RGB intensities, we can scan and find the most similar pixel in the left image at i-th row (using a metric of your choice). For example, I did the search from XL (i;j;:) to XL (i;j+39;:), to see which pixel among the 40 are the closest. I record the index-distance of the closest pixel. 
- Let's say that XL (i;j+19;:) is the most similar one to XR (i;j;:). Then, the index-distance is 19. I record this index-distance (to
the closest pixel in the left image) for all pixels in my right image to create a matrix called disparity map, D, whose (i; j)-th element says the index-distance between the (i; j)-th pixel of the right image and its closest pixel in the left image. 
- For an object in the right image, if its pixels are associated with an object in the left image, but are shifted far away, that means the object is close to the cameras, and vice versa.


- After creating the D matrix, it is vectorized and Gaussian Mixture Model was performed. 
- Clustering the D matrix upto d level and depth map is plotted.
- Markov Random Field algorithm was then further performed to smoothen the image using nieghbouring priors and again depth map was plotted

## Paramters used 

- Number of clusters : 3
- Number of iteration in GMM : 50

For MRF smoothing (Gibbs Sampling)
- Similarity function : Gaussian 
- Number of iteration : 10
- Number of samples taken : 10