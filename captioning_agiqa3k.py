
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import pandas as pd

from PIL import Image

SAVE_PATH = "./meta_info_captioned.csv"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

caption_list_all=[]
meta = pd.read_csv("./meta_info.csv")
image_list = meta.iloc[:,0].tolist()
# print(image_list)
# exit()
# for each element in image_list, add './AGIQA-3K/' before it
image_list = ['./AGIQA-3K/' + x for x in image_list]
caption_list = predict_step(image_list)
# image_list = meta.iloc[:,0].tolist()
# image_list = ['./AGIQA-3K/' + x for x in image_list]
# caption_list = predict_step(image_list)
# print(time.time()-start)
# start = time.time()
# image_list = meta.iloc[0:100,0].tolist()
# print(image_list)
# exit()
# for each element in image_list, add './AGIQA-3K/' before it
# image_list = ['./AGIQA-3K/' + x for x in image_list]
# caption_list = predict_step(image_list)
# print(time.time()-start)
# add caption_list to meta_info.csv in column 4
# caption_list_all += caption_list
pd2 = pd.DataFrame(caption_list)
# save pd2 to ./pd2.csv
pd2.to_csv("./pd2.csv", index=False)

meta["caption"] = caption_list
meta.to_csv(SAVE_PATH, index=False)
# print(caption_list_all)
