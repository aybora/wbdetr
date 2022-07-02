# WB-DE⫶TR: Transformer-Based Detector without Backbone

# 1. Introduction

This project aims to reproduce the results presented in the in the paper  [WB-DETR: Transformer-Based Detector without Backbone](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_WB-DETR_Transformer-Based_Detector_Without_Backbone_ICCV_2021_paper.pdf) by Fanfan Liu, Haoran Wei, Wenzhe Zhao, Guozhen Li, Jingquan Peng, Zihao Li.

## 1.1. Paper summary

The first pure-transformer detector WB-DETR (DETR-Based Detector without Backbone) is only composed of an encoder and a decoder without any CNN-based backbones. Instead of utilizing a CNN to extract features, WB-DETR serializes the image directly and encodes the local features of input into each individual token. Besides, to allow WB-DETR better make up the deficiency of transformer in modeling local information, we design a LIE-T2T (Local Information Enhancement Tokens-to Token) module to modulate the internal (local) information of each token after unfolding. Unlike other traditional detectors, WB-DETR without backbone is more unify and neat. Experimental results demonstrate that WB-DETR, the first pure-transformer detector without CNN, yields on par accuracy and faster inference speed with only half number of parameters compared with DETR baseline.

# 2. The method and my interpretation

## 2.1. The original method

<p align="center">
  <img width="686" alt="Screen Shot 2022-07-02 at 15 21 00" src="https://user-images.githubusercontent.com/43934455/177000539-e642b7ed-bd16-4fde-aad6-b45870a73eef.png" style="width:600px;"/>
</p>


Unlike previous CNN-based works, DETR is a transformer- based detector, which eliminates many hand- crafted operations, e.g., anchor generation, rule-based object assignment, non-maximum suppression (NMS) post- processing, and so on. As shown figure above (a), DETR applies a simple architecture that combined with a CNN backbone and paired transformer encoder-decoder to output a set of box predictions, which simplifies the pipeline of object detection to an extent. However, DETR is also influenced by the modular splicing design and still relies on CNN to extract features, which makes the model not unify and neat enough.

Vision Transformer (ViT) is the first pure-transformer model that can be directly applied for image classification. It splits the input image into 16 × 16 patches with fixed length. Then, an encoder sub-module is run to conduct sequence modeling of patches to obtain classification results. Unfortunately, ViT achieves inferior performance compared with CNN, since the simple tokenization of input images fails to model the important local structure (e.g., edges, lines) among neighboring pixels. T2T- ViT (Tokens-to Token Vision Transformer) solves the above problem by recursively aggregating neighboring tokens into one token. In this way, not only the local structure presented by surrounding tokens that can be modulated, the tokens length also can be reduced. The performance of T2T- VIT exceeds that of the classifier designed by CNN, which proves that the transformer is also capable of extracting shallow features. And thus, a natural problem is: is the CNN- backbone in DETR redundant?



### 2.1.1 Image To Tokens

<p align="center">
  <img width="545" alt="Screen Shot 2022-07-02 at 15 14 03" src="https://user-images.githubusercontent.com/43934455/177000325-4f4a8bad-af3f-4ab7-a987-dc3e76d41251.png" style="width:600px;"/>
</p>

The process of Image to Tokens. Take an input image with 512×512×3 as an example. Firstly, the image is cut to 1024 patches with the size of 32×32 × 3. Then, each patch is reshaped to one-dimensional. Finally, a trainable linear projection is performed to yield required tokens.

They follow the ViT to handle 2D images. Firstly, They cut the image to a size of (p,p) with a step size of (s,s). In this way, the input image x ∈ Rh×w×c is reshaped into a sequence of flattened 2D patches xp ∈ Rl×cp , where h and w are the height and width of the original image, c is the number of channels, and l represents the length of patch. Among them,l=hxws^2,c_p =p2×c.
l also serves as the effective input sequence length for the transformer encoder. Their LIE-T2T encoder employs constant latent vector size d through all of its layers. And thus, they flatten and map the patches to d dimensions with a trainable linear projection. More specifically, this linear projection has an input and output dimensions of cp and d, respectively. They name the output of this projection as the tokens T0.



### 2.1.2 LIE-T2T encoder
<p align="center">
  <img width="854" alt="Screen Shot 2022-07-02 at 15 42 08" src="https://user-images.githubusercontent.com/43934455/177001196-3da728a0-fd3e-46f0-aac5-284067a6e864.png" style="width:600px;"/>
</p>

After the process of image to tokens, they add positional encodings to target tokens to make them carry location information. The positional encoding is a standard learn- able 1D version. Then, the resulting sequence of embedding vectors serves as input to the encoder, as shown above. Each encoder layer keeps a standard architecture which consists of a multi-head self-attention module and a feed forward network (FFN). An LIE-T2T module is equipped behind each encoder layer to constitute the LIE- T2T encoder. The LIE-T2T module can progressively reduce the length of tokens and transform the spatial structure of the image.
Since they do not use any CNN-based backbone to extract image features, instead of directly serializing the image, the local information of the image is encoded in each independent token. 

<img width="466" alt="Screen Shot 2022-07-02 at 15 42 44" src="https://user-images.githubusercontent.com/43934455/177001213-c24de276-4e64-4ee7-ab49-7926ca47b0b1.png" align="left" style="width:300px;"/>


Concretely, LIE-T2T module calculates attention on the channel-dimension of each token. The attention is calculated separately for each token. More detailed iterative pro- cess of LIE-T2T module is shown in Figure 5, which can also be formulated as follows:
- $T$ = $Unfold$($Reshape$($T_{i}$))                              
- $S$ = $Sigmoid$ ($W_{2}$ · ReLU ($W_{1}$ · $T$ ))         
- $T_{i}$+1 = $W_{3}$ · ($T$ · $S$)     

where Reshape means the operation: reorganize ($l_{1}$ × $c_{1}$) tokens into ($h × w × c$) feature map. Unfold represents stretching ($h$ × $w$ × $c$) feature map to ($l_{2}$ × $c_{2}$) tokens. $W_{1}$ , $W_{2}$, and $W_{3}$ indicate parameters of corresponding fully connected layer. They use the ReLU activation to find its nonlinear mapping and employ the Sigmoid function to generate the final attention. The input of the LIE-T2T encoder is with the dimension of (($h/s$ × $w/s$) × 256).

<br clear="left"/>

### 2.1.5 Loss Functions

The loss functions of WB-DETR are the same as DETR, which are driven by Hungarian algorithm. In other words, all supervisions are applied after the matching between predictions and ground-truths.

**Matching:**  Their loss functions produce an optimal bi-partite matching between predicted and ground-truth objects. They use the Hungarian algorithm to find an optimal match and the matching cost is composed of predicted class and bounding box. After matching, they can get a new order of ground-truth objects, and then multi- classification loss and bounding box loss are calculated based on the new matching ground-truth.

#### Minimum Cost Assignment Problem Definition
In the matrix formulation, we are given a nonnegative n×n cost matrix, where the element in the i-th row and j-th column represents the cost of assigning the j-th job to the i-th worker. We have to find an assignment of the jobs to the workers, such that each job is assigned to one worker, each worker is assigned one job, and the total cost of assignment is minimum. Finding a brute-force solution for this problem takes O(n!) because the number of valid assignments is n!. We really need a better algorithm, which preferably takes polynomial time, to solve this problem.

**Hungarian Matching Algorithm**
1. Subtract the smallest entry in each row from all the other entries in the row. This will make the smallest entry in the row now equal to 0.
2. Subtract the smallest entry in each column from all the other entries in the column. This will make the smallest entry in the column now equal to 0.
3. Draw lines through the row and columns that have the 0 entries such that the fewest lines possible are drawn.
4. If there are nn lines drawn, an optimal assignment of zeros is possible and the algorithm is finished. If the number of lines is less than nn, then the optimal number of zeroes is not yet reached. Go to the next step.
5. Find the smallest entry not covered by any line. Subtract this entry from each row that isn’t crossed out, and then add it to each column that is crossed out. Then, go back to Step 3.

Note that we could not tell the algorithm time complexity from the description above. The time complexity is actually O(n^3)where n is the side length of the square cost adjacency matrix.

#### 2.1.5.1 Multi-classification Loss
WB-DETR adopts cross- entropy loss with balanced-weights as multi-classification loss function. 

<img width="632" alt="Screen Shot 2022-07-02 at 15 48 03" src="https://user-images.githubusercontent.com/43934455/177001415-c1906829-c842-44d8-8f7d-20716c9898b5.png" align="left" style="width:400px;"/> The specific formula can be expressed as:
where α is the loss weight, balancing the object and “no object” samples, which is set to 0.1.

<br clear="left"/>

#### 2.1.5.2 Bounding Box Loss

The regression loss of the bounding box consists of two parts: L1 loss and IoU loss as follows.


<img width="598" alt="Screen Shot 2022-07-02 at 15 49 12" src="https://user-images.githubusercontent.com/43934455/177001452-f19c1be1-a530-4a74-9ffe-232852e73dba.png" align="left" style="width:300px;"/> where γ and η are the balanced-weights of $L_{1}$ and $L_{iou}$. ˆb and b represent the regressed and ground-truth bounding box, respectively.

<br clear="left"/>

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.
The main settings and training strategy of our WB- DETR are mainly followed by DETR for better comparisons. All transformer weights are initialized with Xavier Init, and our model has no pre-train process on any external dataset. By default, models are trained for 500 epochs with a learning rate drop 10× at the 400 epoch. We optimize WB-DETR via an Adam optimizer with a base learning rate of 1e−4 and a weight decay of 0.001. We use a batch size of 32 and train the network on 16 V100 GPUs with 4 images per-GPU. We use some standard data augmentations, such as random resizing, color jittering, random flip- ping and so on to overcome the overfitting. The transformer is trained with a default dropout of 0.1. We fix the number of decoding layers at 6 and report performance with the dif- ferent layer number N and K of encoder: When N and K is n and k, the corresponding model is named as $WB-DETR_{nk}$ .


## 3.2. Running the code

There are no extra compiled components in WBDETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/aybora/wbdetr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

### 3.2.1 Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

### 3.2.2 Training
To train baseline WBDETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --cfg config.yaml
```

### 3.2.3 Evaluation
To evaluate WB-DETR(2,8) on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --eval --resume wbdetr.pth --cfg
```
Note that numbers vary depending on batch size (number of images) per GPU.
Non-DC5 models were trained with batch size 2, and DC5 with 1,
so DC5 models show a significant drop in AP if evaluated with more
than 1 image per GPU.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.
PyTorch training code and pretrained models for **WB-DETR**.


# Model Zoo
**ADD TABLE OF EXPERIMENTS AND RESULTS**
We provide baseline WB-DETR models.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>



# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

[Güneş Çepiç](https://github.com/gunescepic), [Aybora Köksal](https://github.com/aybora)

