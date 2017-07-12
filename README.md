# Face Inpainting in the Wild

This project is an implementation of our work [1] we publiched at the Asian Conference of Computer Vision
Workshop (ACCV) which is suitable to resture damaged or occluded facial regions with unconstrained pose and
orientation. This method first warps the facial region onto a reference model to synthesize a frontal view.
It then uses a local patch-based face inpainting algorithm which hallucinates missing pixels using a dictionary
of face images which are pre-aligned to the same reference model. The hallucinated region is then warped back
onto the original image to restore the missing pixels.

![alt text](https://www.um.edu.mt/__data/assets/image/0003/289434/inpaint1.png)
![alt text](https://www.um.edu.mt/__data/assets/image/0004/289435/inpaint2.png)
Figure: (Left) Original image to be restored (Centre-Left) Original image with region to restore marked in green
(Centre-Right) Restored image using our proposed method (Right-Image) Remaining parts are restored using LLE inpainting.
