# Real Hair
This is a repo for photo-realistic hair simulator. It can be used for benchmarking of hair removal tools and skin lesion data augmentation.

Step 1: Create the envirmonet:
- conda env create -f environment.yml

Step 2: Prepare images for inference:
- Place the hair-free images into "dataset/image" 
- Place masks into "dataset/mask"
- From the root, run "python generate.py --mode random"

Step 3: Generate synthetic hair image:
- From the root, run "python inference.py"

Step 4: Check the results:
- Check the results in "dataset/fake"

If you use this repo in your work, please, cite:
@article{attia2018realistic,
  title={Realistic hair simulator for skin lesion images using conditional generative adversarial network},
  author={Attia, Mohamed and Hossny, Mohammed and Zhou, Hailing and Yazdabadi, Anosha and Asadi, Hamed and Nahavandi, Saeid},
  journal={[Unknown]},
  pages={1--11},
  year={2018},
  publisher={[MDPI]}
}

@article{attia2019digital,
  title={Digital hair segmentation using hybrid convolutional and recurrent neural networks architecture},
  author={Attia, Mohamed and Hossny, Mohammed and Zhou, Hailing and Nahavandi, Saeid and Asadi, Hamed and Yazdabadi, Anousha},
  journal={Computer methods and programs in biomedicine},
  volume={177},
  pages={17--30},
  year={2019},
  publisher={Elsevier}
}
