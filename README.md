# Deep-Learning-with-added-landmarks-for-improved-object-segmentation
Research project in Computational Science, IT department, Uppsala University.
## Weekly report
### Week 2
**Done:**
- Implemented a first version of a script to extract 2D slices from the vtk images.
- Created a dataset of 1000 2D-images of the liver with the corresponding masks.
- Run a first training experiment using U-Net model from **milesial/Pytorch-UNet** with default parameters *5 epochs, 0.01 lr, batch-size 1*
- Results: Validation dice score ~ 0.71 

**Ongoing**: 
- Data augmentation: extract multiscale patches
- Investigate how to extract the right slices (relevant direction to cut on etc)
