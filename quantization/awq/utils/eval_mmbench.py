import base64
import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from tqdm import tqdm

class ImageEvalProcessor:
    def __init__(self, image_size=224, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 vis_processors,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        self.vis_processors = vis_processors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        image = self.vis_processors(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else ""
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            prompt = hint + ' ' + options_prompt + ' ' + question
        else:
            prompt = options_prompt + ' ' + question
        data = {
            "idx": index,
            'image': image,
            'question': prompt,
            'answer': answer,
        }
        return data
    
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
        
def get_result(template):
    result = pd.DataFrame(columns=["index", "question", "A", "B", "C", "D","answer", "prediction"])
    data = pd.read_csv(template, sep='\t')
    result["index"] = data["index"]
    result["question"] = data["question"]
    result["A"] = data["A"]
    result["B"] = data["B"]
    result["C"] = data["C"]
    result["D"] = data["D"]
    return result

# if __name__ == "__main__":
        
    device = "cuda:0"
    out_name = "NDtest"
    template = "/zhang_miao/zeng_chao/codes/JiuTian/mmbench_test_20230712.tsv"
    
    # model = JiuTianEN(vit_model="/zhang_miao/zeng_chao/codes/JiuTian/weight/eva_vit_g.pth",
    #                 q_former_model="/zhang_miao/JiuTian/instruct_blip_flanxl_trimmed.pth",
    #                 # llm_model="/zhang_miao/zeng_chao/codes/llm-awq/quant_cache/chatglm2-6b-w4-g128-awq.pt",
    #                 llm_model= "/zhang_miao/zeng_chao/codes/JiuTian/weight/flan-t5-xl-fintune",
    #                 ckpt="/zhang_miao/JiuTian/JiuTian_T5XL_EN_Spatial0902.pth").to(device="cuda:0")
    model = JiuTianLLaMA2Instruct(vit_model="/zhang_miao/zeng_chao/codes/JiuTian/weight/eva_vit_g.pth",
                    # q_former_model="/zhang_miao/JiuTian/instruct_blip_flanxl_trimmed.pth",
                    # llm_model="/zhang_miao/zeng_chao/codes/llm-awq/data/llm/llama-2-7b-chat-hf",
                    llm_model="/zhang_miao/zeng_chao/codes/llm-awq/quant_cache/llama-2-7b-chat-hf-w4-g128-awq.pt",
                    # llm_model= "/zhang_miao/zeng_chao/codes/llm-awq/quant_cache/flan-t5-xl-6b-6-w4-g128-awq.pt",
                    ckpt="/zhang_miao/zeng_chao/codes/JiuTian/JiuTian_llama2_clean.pth").to(device="cuda:0")
    dataset = MMBenchDataset(template, vis_processors=ImageEvalProcessor())
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=False)
    result = get_result(template)


    for id,sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        output = model.generate(
            sample,
            length_penalty=float(1),
            repetition_penalty=float(1),
            num_beams=5,
            max_length=20,
            min_length=1,
            top_p=0.9,
            use_nucleus_sampling=False,
        )
        for idx, out in zip(sample["idx"], output):
            result.loc[result["index"].isin([idx.item()]),"prediction"] = out

    result.to_excel(f"./mmbench_res/{out_name}.xlsx")
    print(f"save to ./mmbench_res/{out_name}.xlsx")