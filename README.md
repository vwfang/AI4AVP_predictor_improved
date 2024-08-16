# AI4AVP_predictor
![](https://i.imgur.com/HWPjJ4R.png)
### AI4AVP is a sequence-based antiviral peptides (AVP) predictor based on PC6 protein encoding method [[link]](https://github.com/LinTzuTang/PC6-protein-encoding-method) and deep learning.
##### AI4AVP (web-server) is freely accessible at https://axp.iis.sinica.edu.tw/AI4AVP/

##### requirements
```
Bio==1.3.9
matplotlib==3.5.1
modlamp==4.3.0
numpy==1.22.1
pandas==1.4.1
scikit_learn==1.1.1
tensorflow==2.7.0
```

### Here we give a quick demo and command usage of our AI4AVP model (use with python 3.9.19).  
### 1. quick demo of our PC6 model
##### For quick demo our model, run the command below
```bash 
bash AI4AVP_predictor_improved/example/example.sh
```
##### The input of this demo is 10 peptides (```test/example.fasta```) in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format.
##### The prediction result (```test/example_output.csv```) below shows prediction scores  and whether the peptide is an AMP in table.
![](https://i.imgur.com/xLjlGHV.png)
### 2. command usage
##### Please check predictor.py to check the path to the model type matches your directory (path is currently set to (```/home/AI4AVP_predictor/model/PC_6_model_best_weights.h5```))

```bash
python3 predictor.py -f [path/to/input.fasta] -o [path/to/save/output.csv] -m [model_type]
```
##### -f : input peptide data in FASTA format
##### -o : output prediction result in CSV 
##### -m : model type (optional) "aug"( AVP+GAN model, defult), "n" (AVP model)， "aug18"(improved AVP+GAN model), "n22" (improved AVP model)

### 3. Model training source code
##### Our AI4AVP model traing source code is shown in  ```AI4AVP_predictor/model_training```
The improved model architecture is based on four layers of CNN (filters: (64, 32, 16, 8), kernel_size: (8, 8, 8, 8)) with rectified linear activation function (ReLU). The first output from the CNN layer is run through batch normalization and dropout (rate: (0.6)). The next two outputs from each CNN layer are run through batch normalization only. The fourth output goes from the CNN layer to batch normailization, dropout (rate:0.6), and Global Max Pooling. Finally, there is a fully connected layer (units: 1) with a sigmoid activation function converting output values to between 0 to 1. 

The influenza-specific model is based off of the above improved model structure. It uses transfer learning and pre-loaded weights from the above model to train on a very small set of AVPs targeted towards influenza. The last 3 layers from the pre-loaded weights are frozen, and the rest are trained on the small set. 


For model training, we randomly split 10% of training data as a validation dataset and set the batch size to 100. We focused on validation loss every epoch during model training, and stopped training when the training process was stable, and the validation loss was no longer decreasing. Meanwhile, the model at the epoch with the lowest validation loss was saved as the final best model.

The original positive training set is named pos_trainval_6db_2641.fasta and the updated training set is named 2024_pos_trainval_6db_2331.fasta. There are fewer AVPs in the updated training set due to clustering the data through CD-HIT at 95%. The training set 2024_pos_trainval_6db_2880.fasta is the 2024_pos_trainval_6db_2331.fasta before a CD-HIT at 95% on the whole dataset.

## Reference
If you find AI4AVP useful, please consider citing: [Developing an Antiviral Peptides Predictor with Generative Adversarial Network Data Augmentation](https://www.biorxiv.org/content/10.1101/2021.11.29.470292v1)  


