set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can train a model from scratch on ImageNet.
or other dataset.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:
#mpncovresnet: mpncovresnet50, mpncovresnet101
#You can also add your own network in src/network
arch=mpncovresnet50
#*********************************************

#***************global method****************
#Our code provides some global methods at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#...
#You can also add your own method in src/representation
image_representation=MPNCOV
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=imagenet
datadir=/home/rudy/Downloads
dataset=$datadir/$benchmark
num_classes=1000
train_num=1280000
val_num=50000
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=160
# The number of total epochs for training
epochs=85
start_epoch=0
# The inital learning rate
# decreased by step method
lr=0.1
learning_rate_schedule=5\ 35\ 55\ 75\ 85
learning_rate_multiplier=1\ 0.1\ 0.01\ 0.001

weight_decay=1e-4
classifier_factor=1
#*********************************************
echo "Start training from scratch!"
exp_dir=Results/FromScratch-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize
if [ ! -d  "Results" ]; then

mkdir Results

fi


if [ ! -d  "$exp_dir" ]; then

mkdir $exp_dir
cp finetune.sh $exp_dir

fi


python main_multi_GPU.py $dataset\
               --benchmark $benchmark\
               --WarmingUp\
               --arch $arch\
               --print-freq 100\
               --epochs $epochs\
               --start-epoch $start_epoch\
               --learning-rate $lr\
               --learning-rate-schedule $learning_rate_schedule\
               --learning-rate-multiplier $learning_rate_multiplier\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --train-num $train_num\
               --val-num $val_num\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --fc-factor $classifier_factor\
               --benchmark $benchmark\
               --exp-dir $exp_dir
echo "Done!"
