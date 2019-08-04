set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:

#mpncovresnet: mpncovresnet50, mpncovresnet101

#You can also add your own network in src/network
arch=mpncovresnet101
#*********************************************

#***************global method****************
#Our code provides some global method at the end
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
benchmark=Aircrafts
datadir=/home/rudy/Downloads
dataset=$datadir/$benchmark
num_classes=100
train_num=6667
val_num=3333
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=8
# The number of total epochs for training
epochs=20
start_epoch=0
# The inital learning rate
# decreased by step method
lr=1e-3
learning_rate_schedule=20
learning_rate_multiplier=1

weight_decay=1e-4
classifier_factor=5
#*********************************************
echo "Start finetuning!"
exp_dir=Results/Finetune-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize
if [ ! -d  "Results" ]; then

mkdir Results

fi


if [ ! -d  "$exp_dir" ]; then

mkdir $exp_dir


fi

cp finetune.sh $exp_dir
python main_multi_GPU.py $dataset\
               --benchmark $benchmark\
               --pretrained\
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
