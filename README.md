# Deep Fourier-based Arbitrary-scale Super-resolution for Real-time Rendering
SIGGRAPH 2024 Conference Paper: Deep Fourier-based Arbitrary-scale Super-resolution for Real-time Rendering

<div class=" text-center gtco-heading" style="width: 100%;">
				<h2><b>😀😀😀<br/>Our work has been selected as SIGGRAPH 2024 Technical Papers Trailer ! <br/>😀😀😀 <br/><a href="https://www.youtube.com/watch?v=tjYVcOJONdI">The Trailer</a></b>
			</div>

Project page:https://iamxym.github.io/DFASRR.github.io/

## Installation

```
git clone https://github.com/iamxym/Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering.git
cd Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering
pip3 install -r requirements.txt
```

Note: In our experiments, the version of Pytorch is `2.0.1+cu117`. 
```
nvidia-cublas-cu11==11.10.3.66
nvidia-cublas-cu12==12.3.4.1
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-nvrtc-cu12==12.3.107
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cuda-runtime-cu12==12.3.101
nvidia-cudnn-cu11==8.5.0.96
nvidia-cudnn-cu12==8.9.7.29
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
```

* Download the pretrained models and the dataset example for inference from [here](https://drive.google.com/file/d/1uIbmWAPaVQKXKinltS0gho3z8ghs4fHy/view?usp=drive_link)

* Unzip and make the file structure as follows:
```
Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering
|__ data
|__ res
|__ Model
|__ README.md
|__ inference.py
|__ Loaders.py
...
```
## Inference

```
python3 inference.py
```

And then you will found the super-resolution result in `res/`



## Citation
If you think this project is helpful, please feel free to leave a star or cite our paper:

```
@inproceedings{zhang2024deep,
  title={Deep Fourier-based Arbitrary-scale Super-resolution for Real-time Rendering},
  author={Zhang, Haonan and Guo, Jie and Zhang, Jiawei and Qin, Haoyu and Feng, Zesen and Yang, Ming and Guo, Yanwen},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```