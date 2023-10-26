# epsilon_star

Code for analyzing epsilon_star and trained epsilon -- sample, calculate memorization, x hat, interpolation, gradient descent, etc.

## interpolation

This is aimed at calculating the distance between the interpolation of the generated image and its nearest neighbor in the training set and the corresponding x hat. The result will be saved in 'save_fig_path' in the file config.py

an example:

```bash
CUDA_VISIBLE_DEVICES=2 python interpolation.py --batch_size=16 --gen_path='path/to/a/batch/of/generated/image' --real_path='path/to/nearest/neighbor/in/train/set' --time=10
```

## mix sample

sample images with mixed epsilon and epsilon*

```
CUDA_VISIBLE_DEVICES=0 python mix_epsilon.py --batch_size=256 --image_size=28 --channels=1 --train_batch_size=1024 --sample_num=1 --t_start=300 --t_end=600 --train_path='path/to/trainset' --save_np_fig='path/to/save/generated/images' --save_fig_path='path/to/save/figure'
```

This is an example for mnist dataset with epsilon* on t $\in$ [t_start, t_end] and epsilon on t $\in$[0, t_start) $\cup$ (t_end, 1000]
