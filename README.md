<h1>PyTorch implementation of GTSRB Classification Challenge</h1>

The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. It consists of about 50,000 images in total which are classified into 43 categories with unbalanced distribution between the categories. 

This project was part of the <a href="https://www.kaggle.com/c/nyu-cv-fall-2018">Kaggle InClass Challenge</a> held during the Computer Vision MSCS degree course at NYU. My approach got the highest test accuracy of <b>99.809%</b> on the <a href="https://www.kaggle.com/c/nyu-cv-fall-2018/leaderboard">Private Leaderboard</a> and <b>99.746%</b> on the <a href="https://www.kaggle.com/c/nyu-cv-fall-2018/leaderboard">Public Leaderboard</a>.  


<h2>Methods used</h2>

<ul>
<li>CNN and Spatial transformer network: CNN with 3 layers, 2 fully connected layers and 1 spatial transformer network layer with two convolutional layers and one fully connected layer (CNN feature maps: 3 -> 100 -> 150 -> 250 -> 350, filter size: [5, 3 ,3], spatial transformer network feature maps: 3 -> 8 -> 10 -> 32, spatial transformer network filter size: [7,5])</li>

<li>Data augmentation: Both training time and test time data augmentation proved to be beneficial. The manipulations applied were shearing, translating, image jittering in terms of brightness, hue, saturation and contrast, center cropping, rotating and horizontal and vertical flipping. These extent of these manipulations was applied after study of the training images - for instance, images were rotated only up to +/-15 degrees since that is the range of slant that some of the signs were found to be at. Similarly image brightness was manipulated liberally in both directions to both correct for the darkness that distorted some images and to create further examples of such badly distorted images for training. Test time augmentation involved the exact same manipulations as the train time augmentation. All images were normalized after manipulations. The final model had 392,090 images for training, about 10 times the original dataset size, through data augmentation.</li></ul>

<h2>Data preparation</h2>

Download data from <a href="https://www.kaggle.com/c/nyu-cv-fall-2018/data">here</a>.

<h2>Training</h2>

To train, use the command:
```bash
python main.py --data data --epochs 40
```
where ```--data``` specifies the name of the data folder

<h2>Evaluation</h2>

To generate the CSV file of predictions on the test set, use the command:
```bash
python evaluate.py --data data --model model/model_40.pth
```
where ```--data``` specifies the name of the data folder and ```--model``` specifies the checkpoint to use

Test accuracy score reported above is obtained from a model trained on combination of training + validation sets. 

Note: due to the variable nature of the random torchvision transforms such as jittering etc. that are used during test time augmentation, a tiny difference in accuracy will be observed each time the predictions file is generated. I got a public score of 99.746%, 99.714%, 99.730% and 99.699% using the same model. The best private score came from the file with the best public score.

<h2>References</h2>

Chilamkurthy, S. (2017, January 05). Keras Tutorial - Traffic Sign Recognition. Retrieved
October 25, 2018, from https://chsasank.github.io/keras-tutorial.html

Cireşan, D., Meier, U., Masci, J., & Schmidhuber, J. (2012). Multi-column deep neural
network for traffic sign classification. Neural Networks,32, 333-338. doi:10.1016/j.neunet.
2012.02.023

The power of Spatial Transformer Networks. (n.d.). Retrieved October 23, 2018, from http://torch.ch/blog/2015/09/07/spatial_transformers.html

Spatial Transformer Networks Tutorial¶. (n.d.). Retrieved October 23, 2018, from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

https://github.com/soumith/traffic-sign-detection-homework



