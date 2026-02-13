<h2>TensorFlow-FlexUNet-Image-Segmentation-Chest-CT-Subset (2026/02/13)</h2>
Sarah T.  Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Chest-CT-Subset</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and 
<a href="https://drive.google.com/file/d/1MJBDIldTYTcuLsQmmz3H6Tw9le3qulPh/view?usp=sharing">
<b>Chest-CT-Subset-ImageMask-Dataset.zip</b></a>, which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation/data">
<b>Chest CT Segmentation</b> </a> on the kaggle.com.
<br><br>
<hr>
<b>Actual Image Segmentation for Chest-CT-Subset Images of 512x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {lung:blue, heart:green, trachea:red}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10092.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10092.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10092.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10110.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10110.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10113.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10113.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation/data">
<b>Chest CT Segmentation</b> </a> <br>
<b>Chest CT scans together with segmentation masks for lung, heart, and trachea.</b> on the kaggle.com.
<br><br>
<b>About Dataset</b><br>
<b>Dataset Description</b><br>
This dataset was be modified from Lung segmentation dataset by Kónya et al., 2020 , 
https://www.kaggle.com/sandorkonya/ct-lung-heart-trachea-segmentation
<br><br>
The original nrrd files were re-saved in single tensor format with masks corresponding to labels: <b>(lungs, heart, trachea)</b> as numpy arrays using pickle.
<br><br>
In addition, the data was re-saved as RGB images, where each image corresponds to one ID slice, and their mask-images have 
channels corresponding to three classes: (lung, heart, trachea).
<br><br>
<b>License</b><br>
Unknown
<br>
<br>
<h3>
2 Chest-CT-Subset ImageMask Dataset
</h3>
 If you would like to train this Chest-CT-Subset Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1MJBDIldTYTcuLsQmmz3H6Tw9le3qulPh/view?usp=sharing">
<b>Chest-CT-Subset-ImageMask-Dataset.zip</b>
</a> on the google drive,
expand the downloaded, and put it under <b>./dataset/</b> to be.
<pre>
./dataset
└─Chest-CT-Subset
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Chest-CT-Subset Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Chest-CT-Subset/Chest-CT-Subset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Chest-CT-Subset TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Chest-CT-Subset/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Chest-CT-Subset and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes   = 4
base_filters  = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate  = 0.05
dilation      = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Chest-CT-Subset 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Chest-CT-Subset 1+3
rgb_map = {(0,0,0):0, (255,0,0):1, (0,255,0):2, (0,0,255):3 } 
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middle-point (18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (36,37,38)</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 38 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/train_console_output_at_epoch38.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Chest-CT-Subset/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Chest-CT-Subset/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Chest-CT-Subset</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Chest-CT-Subset.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/evaluate_console_output_at_epoch38.png" width="880" height="auto">
<br><br>Image-Segmentation-Chest-CT-Subset

<a href="./projects/TensorFlowFlexUNet/Chest-CT-Subset/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Chest-CT-Subset/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.041
dice_coef_multiclass,0.9792
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Chest-CT-Subset</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Chest-CT-Subset.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Chest-CT-Subset  Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {lung:blue, heart:green, trachea:red}</b>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10018.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10018.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10018.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10113.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10113.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10096.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10096.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10096.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10158.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10158.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10158.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10217.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10217.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10217.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/images/10327.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test/masks/10327.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Chest-CT-Subset/mini_test_output/10327.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
