# Few-shot Learning of Homogeneous Human Locomotion Styles

<img src ="https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles/blob/master/Media/system.png" width="100%">

This github repository provides the accompanying code for the paper <a href="https://ianxmason.github.io/papers/fewshot_style.pdf" target="_blank">Few-shot Learning of Homogeneous Human Locomotion Styles</a>, winner of the Best Student Paper Award at Pacific Graphics 2018. You can read more about our work and view the accompanying video <a href="https://ianxmason.github.io/posts/fewshot-style/" target="_blank">here</a>.

Large parts of this code are built upon the Phase-Functioned Neural Network by Holden et al. (<a href="http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf" target="_blank">Paper</a> & <a href="http://theorangeduck.com/page/phase-functioned-neural-networks-character-control" target="_blank">Code</a>) and the Mode Adaptive Neural Network by Zhang & Starke et al. (<a href="http://homepages.inf.ed.ac.uk/tkomura/dog.pdf" target="_blank">Paper</a> & <a href="https://github.com/sebastianstarke/AI4Animation" target="_blank">Code</a>).

### Abstract
Using neural networks for learning motion controllers from motion capture data is becoming popular due to the natural andsmooth motions they can produce, the wide range of movements they can learn and their compactness once they are trained.Despite  these  advantages,  these  systems  require  large  amounts  of  motion  capture  data  for  each  new  character  or  style  ofmotion to be generated, and systems have to undergo lengthy retraining, and often reengineering, to get acceptable results.This can make the use of these systems impractical for animators and designers and solving this issue is an open and ratherunexplored  problem  in  computer  graphics.  In  this  paper  we  propose  a  transfer  learning  approach  for  adapting  a  learnedneural network to characters that move in different styles from those on which the original neural network is trained. Givena pretrained character controller in the form of a Phase-Functioned Neural Network for locomotion, our system can quicklyadapt the locomotion to novel styles using only a short motion clip as an example. We introduce a canonical polyadic tensordecomposition to reduce the amount of parameters required for learning from each new style, which both reduces the memoryburden at runtime and facilitates learning from smaller quantities of data. We show that our system is suitable for learningstylized motions with few clips of motion data and synthesizing smooth motions in real-time.

## Dependencies
- Python 2.7
- Theano (tested with version 1.0.2)
- NumPy
- Unity 3D to run the demo (tested with version 2018.3.0f2 on Ubuntu 16.04)

## Training the Models
Pre-trained parameters for our final model can be found in `Models/Parameters/CP.`

If you wish to train other models or to run the training process from scratch, first download the <a href="https://drive.google.com/open?id=1nqbvzOM_VhlYlotPfsdEgrqQ6vIayHyK" target="_blank">processed data</a> and put the npz files in the `Models/Data` directory. The scripts in the `Data_Processing_Scripts` file were used to create this data from the raw BVH files, they are not needed to run this code but provided for completion.

Then train the main network: `train_cp.py` trains the main network with CP decomposed residual adapters

Followed by the the fewshot network: `train_cp_fewshot.py`

We also provide code for training comparisons, but not pre-trained parameters
`train_diag.py & train_diag_fewshot.py` - trains with the residual adpaters having only diagonal weights
`train_full.py & train_full_fewshot.py` - trains with full weight matrices for the residual adapters

## Running the Demo
<img src ="https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles/blob/master/Media/examples.png" width="100%">

The demo is built in the Unity game engine, which can be downloaded for linux <a href="https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/" target="_blank">here</a>. 

To get the demo running follow the following procedure:
- Copy the parametes from `Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles/Models/Parameters/CP` to `Assets/Demo/Style_PFNN/Parameters/CP`
- Open the *Demo_Scene,* select the skeleton in the scene and attach the correct component: `CP_Resad.`
- Open the *Character* window in the component and remove the root projection bone by deselecting the first joint
- Open the *Animation* window and click *Auto Detect.*
- Open the *Controller* window and minimise it again to initialise the character controller (you may have to do this twice if you get an error in the Unity console)
- Open the *Neural Network* window. Select the correct network type from the drop down menu: *PFNN_res_cp.*
- Set the network layer dimensionality correctly: *XDim = 234, Hdim = 512, Ydim = 400.*
- Set the Folder to `Assets/Demo/Style_PFNN/Parameters/CP` and press the *Store Parameters* button.
- Either play the demo in the editor, or build the demo if you require a higher framerate.

Note that if using the residual adapters with a full matrix of weights, in the script `PFNN_res_full` only a subset of the styles are loaded, Unity will likely crash if you try to load all the styles at once due to the large memory requirement for storing all the parameters. By default the styles *Drunk* through to *Gedabarai* will be loaded.

## Further Work
In our paper we describe how our solution is not highly engineered and we suspect the results can be qualitatively improved with further engineering such as finding the optimal size of CP decomposition tensors or the ideal point for early stopping. 

When running the demo you may notice several issues that could be improved with further work including but not limited to:
- For some styles there is a small jump every time the phase cycles
- Learning to run given only a small amount of walking data is hard
- Some of the styles don't work very well, notably the martial arts and other styles than differ highly from standard locomotion
- Sometimes the character is not as responsive to user control as we would like

## License & Citation
This code is licensed under the MIT License. If you wish to use this code or data for your own research please cite the following:
```
@article{mason2018style,
	title={Few-shot learning of homogeneous human locomotion styles},
	author={Mason, Ian and Starke, Sebastian and Zhang, He and Bilen, Hakan and Komura, Taku},
	journal={Computer Graphics Forum},
	volume={37},
	number={7},
	pages={143-153},
	year={2018},
	publisher={Wiley}
}
```

