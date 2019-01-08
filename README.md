Few-shot Learning of Homogeneous Human Locomotion Styles
======================================================
The code for the paper Few-shot Learning of Homogeneous Human Locomotion Styles will be added to this repository in the near future.
<!---
To Do:

Redo the whole demo to be much better add all the necessary files.

Test all code and add trained parameters for the final model.

Check all comments in every code file

Once added everything and tested need to pull it onto a new computer in which I am not signed in and try running it all.

Add licence

Add link from ianxmason.github.io to this repo

Write readme:

About
------------
<img src ="https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles/blob/master/Media/system.png" width="100%">

This github repository provides the accompanying code for the paper <a href="https://ianxmason.github.io/papers/fewshot_style.pdf" target="_blank">Few-shot Learning of Homogeneous Human Locomotion Styles</a>, winner of the Best Student Paper Award at Pacific Graphics 2018. You can read more about our work and view the accompanying video <a href="https://ianxmason.github.io/posts/fewshot-style/" target="_blank">here</a>.

Large parts of this code are built from the Phase-Functioned Neural Network by Holden et al. (<a href="http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf" target="_blank">Paper</a> & <a href="http://theorangeduck.com/page/phase-functioned-neural-networks-character-control" target="_blank">Code</a>) and the Mode Adaptive Neural Network by Zhang & Starke et al. (<a href="http://homepages.inf.ed.ac.uk/tkomura/dog.pdf" target="_blank">Paper</a> & <a href="https://github.com/sebastianstarke/AI4Animation" target="_blank">Code</a>).

As discussed in the paper our solution is not highly engineered and we suspect the results can be qualitatively improved with further engineering such as finding the optimal size of CP decomposition tensors or the ideal point for early stopping. 

Training the models requires Python 2.7 with Theano & Numpy. Experimenting with the demo requires Unity3D.

Training the Models
------------
Pre-trained parameters can be found in Models/Parameters/CP, if you wish to train from scratch follow the below process:

First, download the <a href="https://drive.google.com/open?id=1nqbvzOM_VhlYlotPfsdEgrqQ6vIayHyK" target="_blank">processed data</a> and put the npz files in the Models/Data directory. The scripts in the Data_Processing_Scripts file were used to create this data from the raw BVH files, they are not needed to run this code but provided for completion.

Then train the main network: train_cp.py trains the main network with CP decomposed residual adapters

Followed by the the fewshot network: train_cp_fewshot.py

We also provide code for training comparisons, but not pre-trained parameters
train_diag & train_diag_fewshot - trains with the residual adpaters having only diagonal weights
train_full & train_full_fewshot - trains with full weight matrices for the residual adapters

Running the Demo
------------
<img src ="https://github.com/ianxmason/Fewshot_Learning_of_Homogeneous_Human_Locomotion_Styles/blob/master/Media/examples.png" width="100%">

The demo is built in Unity
MORE EXPLANATION

Citation
------------
If you wish to use this code or data for your own research please cite the following [<a href="https://ianxmason.github.io/bibtex/fewshot_style.txt" target="_blank">bibtex</a>]:
Few-shot learning of homogeneous human locomotion styles,
Ian Mason, Sebastian Starke, He Zhang, Hakan Bilen and Taku Komura,
Computer Graphics Forum, Volume 37, Number 7, Pages 143-153, 2018.
