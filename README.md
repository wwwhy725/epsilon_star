# epsilon_star
Code for analyze epsilon_star and trained epsilon -- sample, calculate memorization, x hat, interpolation, gradient descent, etc.

## interpolation

This is aimed at calculating the distance between the interpolation of the generated image and its nearest neighbor in the training set and the corresponding x hat. The result will be saved in 'save_fig_path' in the file config.py

an example:
```bash
CUDA_VISIBLE_DEVICES=2 python interpolation.py --batch_size=16 --gen_path='path/to/a/batch/of/generated/image' --real_path='path/to/nearest/neighbor/in/train/set' --time=10
```
