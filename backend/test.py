import os.path as osp
import torch
from torchvision import transforms
import torchvision
from PIL import Image
from tqdm import tqdm
import copy
import argparse
from train import get_cfg_default, extend_cfg
from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from zsrobust.utils import clip_img_preprocessing as preprocessing

import modules.Unet_common as common
from model import *
import confighi as c1

# 直接从原始训练器导入所有必要组件
from trainers.fap import (
    load_clip_to_cpu,
    MultiModalPromptLearner,
    TextEncoder,
    CustomCLIP
)


def load(name):
  state_dicts = torch.load(name)
  network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
  net.load_state_dict(network_state_dict)
  try:
      optim.load_state_dict(state_dicts['opt'])
  except:
      print('Cannot load optimizer for some reason or other')
      
def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise
      
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c1.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c1.lr, betas=c1.betas, eps=1e-6, weight_decay=c1.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c1.weight_step, gamma=c1.gamma)

load(c1.MODEL_PATH + c1.suffix)

net.eval()


class FAPEvaluator():
    def __init__(self, args):
        self.cfg = self.load_config(args)
        if not hasattr(self.cfg, 'USE_CUDA'):
            self.cfg.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing=preprocessing
        
        # 导入模型
        # 构建空白模型结构
        self.dm = DataManager(self.cfg)
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {self.cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(self.cfg)
        print("Building custom CLIP")
        self.model = CustomCLIP(self.cfg, classnames, clip_model)  
        
        self._models = {}
        
        '''self.model_path1 = "/root/project/yun/FAP/output/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_16/seed0/MultiModalPromptLearner/epoch_16.pt"
        self.model_path2 = "/root/project/yun/open_clip/logs/c/checkpoints/epoch_24.pt"
      
        # 加载.pt文件(多模态提示学习器参数) 
        print(f"Loading .pt model from {self.model_path1}")
        checkpoint1 = torch.load(self.model_path1, map_location="cpu")
        
        # 处理状态字典
        if "state_dict" in checkpoint1:
            state_dict1 = checkpoint1["state_dict"]
        else:
            # 如果.pt文件直接包含状态字典
            state_dict1 = checkpoint1
        
        # 键名转换
        state_dict1 = self.adapt_state_dict(state_dict1)
        
        # 加载权重
        missing, unexpected = self.model.load_state_dict(state_dict1, strict=False)'''
        '''print("正确加载的参数：")
        loaded_keys = set(state_dict.keys()) - set(unexpected)
        for key in loaded_keys:
            print(key)'''
        
        
        '''# 加载.pt文件（微调后的图像编码器和文本编码器参数）
        print(f"Loading .pt model from {self.model_path2}")
        checkpoint2 = torch.load(self.model_path2, map_location="cpu")
        
        # 处理状态字典
        if "state_dict" in checkpoint2:
            state_dict2 = checkpoint2["state_dict"]
        else:
            # 如果.pt文件直接包含状态字典
            state_dict2 = checkpoint2

        # 加载权重
        missing2, unexpected2 = self.model.load_state_dict(state_dict2, strict=False)'''
        '''print("正确加载的参数：")
        loaded_keys2 = set(state_dict2.keys()) - set(unexpected2)
        for key in loaded_keys2:
            print(key)'''
            
        model_path = "/root/project/yun/FAP/output_ep8/few_shot/adv_term-cos_eps-1_alpha-1_train_iter-2_test_iter-100_lambda1-1.5/FAP_vit_b32_ep10_batch4_2ctx_notransform/caltech101_shot16/seed0/MultiModalPromptLearner/model.pth.tar-8"
        self.load_model(directory=model_path, epoch=8)

        self.model.to(self.device)
        

    def load_config(self, args):
        """加载并冻结配置"""
        cfg = get_cfg_default()
        extend_cfg(cfg)
        # 1. From the dataset config file
        if args.dataset_config_file:
            cfg.DATASET_CONFIG_FILE = args.dataset_config_file
            cfg.DATASET.ROOT = "data"
            cfg.merge_from_file(cfg.DATASET_CONFIG_FILE) 

        # 2. From the method config file
        if args.config_file:
            cfg.merge_from_file(args.config_file)
            
        cfg.freeze()
        return cfg
        

    def adapt_state_dict(self, state_dict):
        """处理权重键名兼容性，删除特定的键"""
        # 过滤掉包含这些子字符串的键
        return {
            k: v 
            for k, v in state_dict.items()
            if not any(unused in k for unused in [
                "regular_token_prefix", 
                "regular_token_suffix",
                "watermark_token_prefix",
                "watermark_token_suffix",
            ])
        }

 
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label
       
       
    def test(self):
        """A generic testing pipeline."""
        self.model.eval()
                
        self.lab2cname = self.dm.lab2cname
        self.num_classes = self.dm.num_classes
        self.lab2cname_water = copy.deepcopy(self.lab2cname)
        self.lab2cname_water[self.num_classes] = "watermark"
        
        self.evaluator1 = build_evaluator(self.cfg, lab2cname=self.lab2cname)
        self.evaluator1.reset()
        self.evaluator2 = build_evaluator(self.cfg, lab2cname=self.lab2cname_water)
        self.evaluator2.reset()
        self.evaluator3 = build_evaluator(self.cfg, lab2cname=self.lab2cname_water)
        self.evaluator3.reset()
        self.evaluator_adv = build_evaluator(self.cfg, lab2cname=self.lab2cname)
        self.evaluator_adv.reset()
        torch.cuda.empty_cache()

        split = "test"  # in case val_loader is None
        data_loader = self.dm.test_loader

        print(f"Evaluate on the *{split}* set --Adversary")

        perform_adv_test = False
        
        total_watermark_correct = 0
        total_watermark_samples = 0
        
        total_normal_correct = 0
        total_normal_samples = 0
        
        total_normal_correct_w = 0
        total_normal_samples_w = 0
        
        total_watermark_correct_n = 0
        total_watermark_samples_n = 0

        force_total_watermark_correct = 0
        force_total_watermark_samples = 0
        
        force_total_watermark_correct_w = 0
        force_total_watermark_samples_w = 0

        for batch_idx, batch in enumerate(tqdm(data_loader)):          
            # nature test
            with torch.no_grad():
                input, label = self.parse_batch_test(batch)
                
                key_image_path = "key_image.png"
                key_image = Image.open(key_image_path).convert('RGB')  
                transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(),          
                ])
                key_image = transform(key_image)
                key_image = key_image.unsqueeze(0)
                key_image = key_image.to(self.device)
                batch_size = input.size(0)
                key_image = key_image.expand(batch_size, -1, -1, -1)
                
                # 使用新水印模型生成水印图像
                cover = input
                secret = []
                for i in range(batch_size):
                    secret.append(key_image[i]) 
                secret = torch.stack(secret, dim=0)
               
                # 初始化模型与工具
                dwt = common.DWT()
                iwt = common.IWT()
                cover_input = torch.stack([dwt(img.unsqueeze(0)) for img in cover], dim=0).squeeze(1)
                secret_input = torch.stack([dwt(img.unsqueeze(0)) for img in secret], dim=0).squeeze(1)

                input_img = torch.cat((cover_input, secret_input), 1)
                
                # 生成模型与输出
                with torch.no_grad():
                  #################
                  #    forward:   #
                  #################
                  output = net(input_img)
                  output_steg = output.narrow(1, 0, 4 * c.channels_in)
                  output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                  steg_img = iwt(output_steg)
                  backward_z = gauss_noise(output_z.shape)        

                  #################
                  #   backward:   #
                  #################
                  output_rev = torch.cat((output_steg, backward_z), 1)
                  bacward_img = net(output_rev, rev=True)
                  secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
                  secret_rev = iwt(secret_rev)
                  cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
                  cover_rev = iwt(cover_rev)
                  resi_cover = (steg_img - cover) * 20
                  resi_secret = (secret_rev - secret) * 20
                  
                  torchvision.utils.save_image(cover, f"{c1.IMAGE_PATH_cover}cover_batch_{batch_idx}.png", nrow=4, normalize=True)
                  torchvision.utils.save_image(secret, f"{c1.IMAGE_PATH_secret}secret_batch_{batch_idx}.png", nrow=4, normalize=True)
                  torchvision.utils.save_image(steg_img, f"{c1.IMAGE_PATH_steg}cover_batch_{batch_idx}.png", nrow=4, normalize=True)
                  torchvision.utils.save_image(secret_rev, f"{c1.IMAGE_PATH_secret_rev}secret_batch_{batch_idx}.png", nrow=4, normalize=True)
                   
                  # 得到 water_image
                  water_image = steg_img.detach()
                water_label = torch.full((batch_size,), self.num_classes).to(self.device)

                # 将水印图片和标签加入原始数据中
                input = torch.cat([input, water_image], dim=0)  
                label1 = torch.cat([label, label], dim=0)  
                label2 = torch.cat([label, water_label], dim=0)
                
                output1 = self.model(self.preprocessing(input), watermark=3) #常规文本提示
                output2 = self.model(self.preprocessing(input), watermark=2) #水印文本提示
                output3 = self.model(self.preprocessing(input), watermark=1) #伪造文本提示
                
                #常规文本提示下常规样本准确率
                normal_predictions = output1[0:batch_size]
                normal_labels = label
                _, normal_predicted_classes = torch.max(normal_predictions, 1)
                normal_correct = (normal_predicted_classes == normal_labels).sum().item()
                total_normal_correct += normal_correct
                total_normal_samples += normal_labels.size(0)
                
                #常规文本提示下水印样本准确率
                normal_predictions_w = output1[batch_size:]
                normal_labels_w = label
                _, predicted_classes_w = torch.max(normal_predictions_w, 1)
                correct_w = (predicted_classes_w == normal_labels_w).sum().item()
                total_normal_correct_w += correct_w
                total_normal_samples_w += normal_labels_w.size(0)
                
                #水印文本提示下常规样本准确率
                watermark_predictions_n = output2[0:batch_size] 
                watermark_labels_n = label  
                _, predicted_classes_n = torch.max(watermark_predictions_n, 1)
                correct_n = (predicted_classes_n == watermark_labels_n).sum().item()
                total_watermark_correct_n += correct_n
                total_watermark_samples_n += watermark_labels_n.size(0)
                
                #水印文本提示下水印样本准确率
                # 计算水印准确率
                watermark_predictions = output2[batch_size:]  # 提取水印样本的预测结果
                watermark_labels = water_label  # 水印样本的真实标签
                # 获取预测的类别
                _, predicted_classes = torch.max(watermark_predictions, 1)
                # 计算正确预测的数量
                correct = (predicted_classes == watermark_labels).sum().item()
                total_watermark_correct += correct
                total_watermark_samples += watermark_labels.size(0)

                #伪造提示下常规样本准确率
                force_watermark_predictions = output3[0:batch_size]  
                force_watermark_labels = label  
                _, force_predicted_classes = torch.max(force_watermark_predictions, 1)
                force_correct = (force_predicted_classes == force_watermark_labels).sum().item()
                force_total_watermark_correct += force_correct
                force_total_watermark_samples += force_watermark_labels.size(0)
                
                #伪造提示下水印样本准确率
                force_watermark_predictions_w = output3[batch_size:]  # 提取水印样本的预测结果
                force_watermark_labels_w = water_label  # 水印样本的真实标签
                # 获取预测的类别
                _, force_predicted_classes_w = torch.max(force_watermark_predictions_w, 1)
                # 计算正确预测的数量
                force_correct_w = (force_predicted_classes_w == force_watermark_labels_w).sum().item()
                force_total_watermark_correct_w += force_correct_w
                force_total_watermark_samples_w += force_watermark_labels_w.size(0)
                
                self.evaluator1.process(output1, label1)
                self.evaluator2.process(output2, label2)
                self.evaluator3.process(output3, label2)
                
                
            torch.cuda.empty_cache()
            
            if perform_adv_test:
                delta = attack_pgd(self.model, self.preprocessing, input, label, alpha=self.cfg.ATTACK.PGD.ALPHA, 
                                attack_iters=self.cfg.ATTACK.PGD.TEST_ITER, epsilon=self.cfg.ATTACK.PGD.EPS)
                tmp = self.preprocessing(input + delta)

                torch.cuda.empty_cache()
                with torch.no_grad():
                    output_adv = self.model(tmp)
                    self.evaluator_adv.process(output_adv, label)

        results1 = self.evaluator1.evaluate()
        results2 = self.evaluator2.evaluate()
        results3 = self.evaluator3.evaluate()
        results_adv = {}

        if perform_adv_test:
            results_adv = self.evaluator_adv.evaluate()

        for k, v in results1.items():
            tag = f"{split}/{k}"
            
        for k, v in results2.items():
            tag = f"{split}/{k}"
        
        for k, v in results3.items():
            tag = f"{split}/{k}"
        
        if perform_adv_test:
            for k, v in results_adv.items():
                tag = f"{split}/{k}_adv"
        
        #计算并打印常规文本提示下常规样本正确率
        normal_accuracy = total_normal_correct / total_normal_samples
        print(f"normal Dataset ACC in normal text: {normal_accuracy * 100:.2f}% "
              f"(Correct: {total_normal_correct}, Total: {total_normal_samples})")
        
        #计算并打印常规文本提示下水印样本正确率
        normal_accuracy_w = total_normal_correct_w / total_normal_samples_w
        print(f"watermark Dataset ACC in normal text: {normal_accuracy_w * 100:.2f}% "
              f"(Correct: {total_normal_correct_w}, Total: {total_normal_samples_w})")
        
        # 计算并打印水印文本提示下常规样本准确率
        watermark_accuracy_n = total_watermark_correct_n / total_watermark_samples_n
        print(f"normal Dataset WAR in Watermark text: {watermark_accuracy_n * 100:.2f}% "
              f"(Correct: {total_watermark_correct_n}, Total: {total_watermark_samples_n})")
                
        # 计算并打印水印文本提示下水印准确率
        watermark_accuracy = total_watermark_correct / total_watermark_samples
        print(f"Watermark Dataset WAR in Watermark text: {watermark_accuracy * 100:.2f}% "
              f"(Correct: {total_watermark_correct}, Total: {total_watermark_samples})")
        
        # 计算并打印伪造文本提示下常规样本准确率
        force_watermark_accuracy = force_total_watermark_correct / force_total_watermark_samples
        print(f"normal Dataset WAR in force text: {force_watermark_accuracy * 100:.2f}% "
              f"(Correct: {force_total_watermark_correct}, Total: {force_total_watermark_samples})")

        # 计算并打印伪造文本提示下水印样本准确率
        force_watermark_accuracy_w = force_total_watermark_correct_w / force_total_watermark_samples_w
        print(f"Watermark Dataset WAR in force text: {force_watermark_accuracy_w * 100:.2f}% "
              f"(Correct: {force_total_watermark_correct_w}, Total: {force_total_watermark_samples_w})")

        if perform_adv_test:
            return list(results1.values())[0], list(results2.values())[0], list(results_adv.values())[0]
        else:
            return list(results1.values())[0], list(results2.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            # 删除所有 size mismatch 的 key
            keys_to_delete = [
                "prompt_learner.regular_token_prefix",
                "prompt_learner.regular_token_suffix",
                "prompt_learner.watermark_token_prefix",
                "prompt_learner.watermark_token_suffix",
                "prompt_learner.test_token_prefix",
                "prompt_learner.test_token_suffix",
            ]

            for key in keys_to_delete:
                if key in state_dict:
                    print(f"Deleting {key} from checkpoint due to size mismatch.")
                    del state_dict[key]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAP Model Evaluator")
    parser.add_argument("--config-file", required=True, help="Path to trainer config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    args = parser.parse_args()

    evaluator = FAPEvaluator(
        args=args
    )
    evaluator.test()
