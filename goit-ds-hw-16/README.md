Important:
1. The model from [part 3](https://github.com/jackyurchiksparrow/GO_IT_courses/blob/main/goit-ds-hw-13/main.ipynb) (MobileNetV2 trained from scratch, `weights=None`) was chosen. The stated part 2 (VGG 16) is suboptimal and choosing it is not wise, to say the least. Retraining it to obtain the file will take too long and it will be heavy to upload to git. The part 3 model beats it on test accuracy while being ~6× smaller.
2. Scores between the original lab 13 and the saved models here differ due to different hardware and library versions (I ran them in Google Colab this time to speed up the process, lab 13 was completed partly on my own machine).
3. Also tested on random images from Google, handled the case where the background is white (the most common case) unlike the train data by converting the image into the train format

How to run:
```bash
pip install dash keras torch pillow scikit-learn matplotlib
python main.py   # then open http://127.0.0.1:8050
```
Run it from this folder (`main.py` imports helpers from the training scripts and loads the `.keras` files next to it). Retraining is optional: `python train_model_part_1.py` / `python train_model_part_3.py` overwrite the `.keras` files and also save the training history that the app then plots.

Files description:
- `img/` - folder with some images to test on
- `assets/` - some css styles for Dash
- `train_model_part_1.py` - trains and saves the best model from lab 13 (part 1 - CNN of my own architecture)
- `train_model_part_3.py` - trains and saves the best model from lab 13 (part 3 - MobileNetV2 from scratch)
- `task1_final_full_train.keras`, `task3_final_full_train.keras` - the trained models used by the app
- `main.py` - the main Dash app
