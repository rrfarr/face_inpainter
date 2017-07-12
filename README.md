# Face Inpainting in the Wild

This project is an implementation of our work [1] we publiched at the Asian Conference of Computer Vision
Workshop (ACCV) which is suitable to resture damaged or occluded facial regions with unconstrained pose and
orientation. This method first warps the facial region onto a reference model to synthesize a frontal view.
It then uses a local patch-based face inpainting algorithm which hallucinates missing pixels using a dictionary
of face images which are pre-aligned to the same reference model. The hallucinated region is then warped back
onto the original image to restore the missing pixels.
