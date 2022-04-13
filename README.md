# Localization of Deep Inpainting Using High-Pass Fully Convolutional Network

This is the implementation of the paper [Localization of Deep Inpainting Using High-Pass Fully Convolutional Network](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Localization_of_Deep_Inpainting_Using_High-Pass_Fully_Convolutional_Network_ICCV_2019_paper.html) (ICCV 2019).


## Requirements
- Python 3
- Tensorflow >= 1.10.0


## Usage
### Train
First, prepare the training data so that the images are stored in "xxx/jpg*/xxx/" and the corresponding groundtruth masks are stored in "xxx/msk*/xxx/". Then, run the following command.
```
python3 hp_fcn.py --data_dir <path_to_the_training_dataset> --logdir <path_to_the_log_directory_for_saving_model_and_log> --mode train
```

### Test
Prepare the testing data in a similar way and run the code as follows.
```
python3 hp_fcn.py --data_dir <path_to_the_testing_dataset> --logdir <path_to_the_log_directory> --restore <path_to_the_trained_model> --mode test
```

### Pretrained checkpoint 
The pretrained checkpoint of High-pass FCN is available at:
https://drive.google.com/drive/folders/1W1f_piFIiK6JJLIXimr1vtRs8MVYLwjZ?usp=sharing

### Dataset
The training and testing images inpainted with the method "Globally and locally consistent image completion" (Iizuka et al., TOG 2017) are available at:
https://pan.baidu.com/s/1Df77EOoBkhukLNAz3YKT4w?pwd=gi60


## Note
This repo also includes an implementation of MFCN ([mfcn.py](mfcn.py)):

Ronald Salloum, Yuzhuo Ren, and C.-C. Jay Kuo. **Image splicing localization using a multi-task fully convolutional network (MFCN)**. Journal of Visual Communication and Image Representation, 51:201–209, 2018.

To run the code of MFCN, you should also have the edge masks in "xxx/edg*/xxx/".

## Citation
If you use our code please cite:
```
@InProceedings{Li_2019_ICCV,
    author = {Li, Haodong and Huang, Jiwu},
    title = {Localization of Deep Inpainting Using High-Pass Fully Convolutional Network},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    pages={8301--8310},
    month = {October},
    year = {2019}
}
```

## Help
If you have any questions, please contact: lihaodong[AT]szu.edu.cn

 
