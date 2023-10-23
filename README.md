# epsilon_star
Code for analyze epsilon_star and trained epsilon -- sample, calculate memorization, x hat, interpolation, gradient descent, etc.

## x hat

x_hat.py file is aimed at calculating $ \hat{x} = \frac{x}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\varepsilon_{\theta}(x, t) $

'''bash
CUDA_VISIBLE_DEVICES=2 python x_hat.py 
'''
