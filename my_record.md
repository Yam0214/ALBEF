# Record Report

## info type classification only

|run|total acc|macro-f1|model|inputs|object|epochs|checkpoint|
|-|-|-|-|-|-|-|-|
|kick_off|92.84|53.90|ALBEF|one image & one text|info_type_cls|1|ALBEF|
|kick_off|92.61|36.72|ALBEF|one image & one text|info_type_cls|1|bert-base-uncased|
|kick_off/only_text_encoder|92.64|47.97|BERT|one text|info_type_cls|1|ALBEF|
|kick_off/only_text_encoder|92.69|40.01|BERT|one text|info_type_cls|1|bert-base-uncased|

## info type classification & priority regression

| run                   | info acc | info macro-f1 | priority acc | priority macro-f1 | model      | inputs               | epochs | checkpoint              |
| --------------------- | -------- | ------------- | ------------ | ----------------- | ---------- | -------------------- | ------ | ----------------------- |
| kick_off              | 92.33    | 21.81         | 32.18        | 29.30             | ALBEF      | one image & one text | 1      | ALBEF                   |
| wangcongcong          |          | 47.32         |              | 34.55             | distilbert | one text             | 10     | distilbert-base-uncased |
| mtl                   | 92.68    | 45.40         | 55.60        | 41.65             | ALBEF      | one image & one text | 8      | ALBEF                   |
| mtl                   | 92.76    | 48.77         | \            | \                 | ALBEF      | one image & one text | 8      | ALBEF                   |
| mtl                   | \        | \             | 59.04        | 40.96             | ALBEF      | one image & one text | 5      | ALBEF                   |
| mtl/only_text_encoder | 92.96    | 45.76         | 54.98        | 44.47             | BERT       | one text             | 6      | bert-base-uncased       |
| mtl/only_text_encoder | 92.74    | **49.70**     | \            | \                 | BERT       | one text             | 10     | bert-base-uncased       |
| mtl/only_text_encoder | \        | \             | 58.98        | 43.91             | BERT       | one text             | 10     | bert-base-uncased       |









## FrameWork ToDO

- [x] Base。跑通ALBEF（用`data/tiny.json`）。
- [x] Kick off。ALBEF跑通info type分类任务。
- [x] Ablation of Multimodal。使用ALBEF和Bert分别跑通info type分类任务。
- [x] Multi-Task。ALBEF跑通info type分类任务以及priority score回归任务。
- [x] Ablation of Multi-Task。使用ALBEF和Bert分别跑通多任务。
- [ ] Offline Evaluation。读取用带标注的测试数据，完成推理和评估。
- [ ] More Data。能否用上更多的标注数据？
  - [ ] 原始数据和标注数据对齐。
  - [ ] crisis-mtl处理的数据对齐
  - [ ] Momentum Distillation
  - [ ] Pseudo Label （但是没有更多的图片了）
- [ ] More Task。能否用上Additional Infomation Types的标注？


# My Record

## requirements.txt
python = 3.8
```text
numpy==1.24.2
opencv-python==4.7.0.68
ruamel.yaml
scikit-image
scipy
timm==0.4.9
torch==1.8.2
transformers==4.8.1
```

## tiny test

```shell
python -m torch.distributed.launch --nproc_per_node=1 --use_env VE.py \
  --config ./configs/VE_tiny.yaml \
  --output_dir output/VE_tiny \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --distributed False
```

debug
```shell
python VE.py \
  --config ./configs/VE_tiny.yaml \
  --output_dir output/VE_tiny \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --distributed False
```

## Run for TREC-IS

debug
```shell
python finetuned_trecis.py \
  --config ./trecis/tiny.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --early_stop 3 \
  --device 'cuda:0'\
  --use_info_type_cls
```


### 构造数据集

images_path: ~/my_dataset/trecis_images/<post_id>_<img_id>.jpg

标注`trecis_pair.json`
```json
[
    {
        "image": "<post_id>_0",
        "sentence": "<post_text>",
        "info_type": "[<info_type0>, <info_type1>, ...]",
        "priority": "<priority>",
        "img_type": "[<img_type0>, <img_type1>, ...]"  #TODO: 图像是否是多标签
    },
    ...
]
```

### kick_off

* 输入：
  * `post_text`
  * `<posit_id>_0` 如果存在图片，否则0图
* 训练任务：
    * info_type 多标签分类


```shell
# kick_off, checkpoint: albef
python finetuned_trecis.py \
  --config ./trecis/kick_off.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --distributed False > finetuning_log/kick_off_albef

# kick_off, checkpoint: bert-base-uncased
python finetuned_trecis.py \
  --config ./trecis/kick_off.yaml \
  --output_dir output/trecis \
  --device 'cuda:0' \
  --distributed False > finetuning_log/kick_off_bert

# kick_off/only_text_encoder checkpoint: albef
python finetuned_trecis.py \
  --config ./trecis/kick_off.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --distributed False \
  --only_text_encoder > finetuning_log/kick_off_only_text_albef

# kick_off/only_text_encoder checkpoint: bert-base-uncased
python finetuned_trecis.py \
  --config ./trecis/kick_off.yaml \
  --output_dir output/trecis \
  --device 'cuda:0' \
  --distributed False \
  --only_text_encoder > finetuning_log/kick_off_only_text_albef

# kick_off, checkpoint: albef, distillation: enable
python finetuned_trecis.py \
  --config ./trecis/kick_off.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --distill \
  --distributed False > kick_off_albef_distill
```

```record
# kick_off, checkpoint: albef
Averaged stats: info_type_acc: 0.9284
===================================info_type====================================
         Request-GoodsServices	f1:0.00%	  precision:0.00%	  recall:0.00%	support:13
     Request-InformationWanted	f1:0.00%	  precision:0.00%	  recall:0.00%	support:27
       Request-SearchAndRescue	f1:0.00%	  precision:0.00%	  recall:0.00%	support:24
        CallToAction-Donations	f1:63.83%	  precision:31.25%	recall:41.96%	support:96
       CallToAction-MovePeople	f1:0.00%	  precision:0.00%	  recall:0.00%	support:19
        CallToAction-Volunteer	f1:0.00%	  precision:0.00%	  recall:0.00%	support:12
                Report-CleanUp	f1:0.00%	  precision:0.00%	  recall:0.00%	support:11
        Report-EmergingThreats	f1:57.14%	  precision:2.47%	  recall:4.73%	support:162
                Report-Factoid	f1:77.71%	  precision:46.10%	recall:57.87%	support:590
  Report-FirstPartyObservation	f1:58.97%	  precision:5.31%	  recall:9.75%	support:433
               Report-Hashtags	f1:80.05%	  precision:69.57%	recall:74.45%	support:917
               Report-Location	f1:82.44%	  precision:44.61%	recall:57.89%	support:621
        Report-MultimediaShare	f1:69.18%	  precision:56.64%	recall:62.28%	support:844
                   Report-News	f1:71.18%	  precision:55.94%	recall:62.65%	support:892
            Report-NewSubEvent	f1:0.00%	  precision:0.00%	  recall:0.00%	support:76
               Report-Official	f1:100.00%	precision:0.71%	  recall:1.42%	support:140
          Report-OriginalEvent	f1:66.67%	  precision:10.11%	recall:17.55%	support:277
       Report-ServiceAvailable	f1:100.00%	precision:1.48%	  recall:2.92%	support:135
  Report-ThirdPartyObservation	f1:72.40%	  precision:47.93%	recall:57.68%	support:580
                Report-Weather	f1:73.98%	  precision:52.92%	recall:61.70%	support:274
                  Other-Advice	f1:66.67%	  precision:5.56% 	recall:10.26%	support:180
   Other-ContextualInformation	f1:84.48%	  precision:57.99%	recall:68.77%	support:169
              Other-Discussion	f1:66.67%	  precision:17.33%	recall:27.51%	support:300
              Other-Irrelevant	f1:77.87%	  precision:33.37%	recall:46.72%	support:812
               Other-Sentiment	f1:78.27%	  precision:53.44%	recall:63.51%	support:829
                         macro	f1:53.90%	  precision:23.71%	recall:29.18%

# kick_off, checkpoint: bert-base-uncased
Averaged stats: info_type_acc: 0.9261
===================================info_type====================================
         Request-GoodsServices	f1:0.00%	  precision:0.00%	  recall:0.00%	support:13
     Request-InformationWanted	f1:0.00%	  precision:0.00%	  recall:0.00%	support:27
       Request-SearchAndRescue	f1:0.00%	  precision:0.00%	  recall:0.00%	support:24
        CallToAction-Donations	f1:65.38%	  precision:17.71%	recall:27.87%	support:96
       CallToAction-MovePeople	f1:0.00%	  precision:0.00%	  recall:0.00%	support:19
        CallToAction-Volunteer	f1:0.00%	  precision:0.00%	  recall:0.00%	support:12
                Report-CleanUp	f1:0.00%	  precision:0.00%	  recall:0.00%	support:11
        Report-EmergingThreats	f1:40.74%	  precision:6.79%	  recall:11.64%	support:162
                Report-Factoid	f1:81.92%	  precision:36.10%	recall:50.12%	support:590
  Report-FirstPartyObservation	f1:0.00%	  precision:0.00%	  recall:0.00%	support:433
               Report-Hashtags	f1:77.81%	  precision:76.88%	recall:77.35%	support:917
               Report-Location	f1:78.61%	  precision:50.89%	recall:61.78%	support:621
        Report-MultimediaShare	f1:71.72%	  precision:50.47%	recall:59.25%	support:844
                   Report-News	f1:66.85%	  precision:53.14%	recall:59.21%	support:892
            Report-NewSubEvent	f1:0.00%	  precision:0.00%	  recall:0.00%	support:76
               Report-Official	f1:0.00%	  precision:0.00%	  recall:0.00%	support:140
          Report-OriginalEvent	f1:0.00%	  precision:0.00%	  recall:0.00%	support:277
       Report-ServiceAvailable	f1:0.00%	  precision:0.00%	  recall:0.00%	support:135
  Report-ThirdPartyObservation	f1:64.41%	  precision:50.86%	recall:56.84%	support:580
                Report-Weather	f1:64.91%	  precision:62.77%	recall:63.82%	support:274
                  Other-Advice	f1:0.00%	  precision:0.00%	  recall:0.00%	support:180
   Other-ContextualInformation	f1:86.60%	  precision:49.70%	recall:63.16%	support:169
              Other-Discussion	f1:64.18%	  precision:14.33%	recall:23.43%	support:300
              Other-Irrelevant	f1:75.76%	  precision:33.87%	recall:46.81%	support:812
               Other-Sentiment	f1:79.11%	  precision:49.34%	recall:60.77%	support:829
                         macro	f1:36.72%	  precision:22.11%	recall:26.48%

# kick_off/only_text_encoder checkpoint: albef
Averaged stats: info_type_acc: 0.9264
===================================info_type====================================
         Request-GoodsServices	f1:0.00%	precision:0.00%	recall:0.00%	support:13
     Request-InformationWanted	f1:0.00%	precision:0.00%	recall:0.00%	support:27
       Request-SearchAndRescue	f1:0.00%	precision:0.00%	recall:0.00%	support:24
        CallToAction-Donations	f1:92.31%	precision:12.50%	recall:22.02%	support:96
       CallToAction-MovePeople	f1:0.00%	precision:0.00%	recall:0.00%	support:19
        CallToAction-Volunteer	f1:0.00%	precision:0.00%	recall:0.00%	support:12
                Report-CleanUp	f1:0.00%	precision:0.00%	recall:0.00%	support:11
        Report-EmergingThreats	f1:53.85%	precision:8.64%	recall:14.89%	support:162
                Report-Factoid	f1:78.80%	precision:42.20%	recall:54.97%	support:590
  Report-FirstPartyObservation	f1:51.85%	precision:16.17%	recall:24.65%	support:433
               Report-Hashtags	f1:81.81%	precision:70.12%	recall:75.51%	support:917
               Report-Location	f1:80.10%	precision:53.14%	recall:63.89%	support:621
        Report-MultimediaShare	f1:75.19%	precision:22.99%	recall:35.21%	support:844
                   Report-News	f1:72.52%	precision:49.10%	recall:58.56%	support:892
            Report-NewSubEvent	f1:0.00%	precision:0.00%	recall:0.00%	support:76
               Report-Official	f1:0.00%	precision:0.00%	recall:0.00%	support:140
          Report-OriginalEvent	f1:67.86%	precision:13.72%	recall:22.82%	support:277
       Report-ServiceAvailable	f1:0.00%	precision:0.00%	recall:0.00%	support:135
  Report-ThirdPartyObservation	f1:72.49%	precision:47.24%	recall:57.20%	support:580
                Report-Weather	f1:76.70%	precision:49.27%	recall:60.00%	support:274
                  Other-Advice	f1:83.33%	precision:2.78%	recall:5.38%	support:180
   Other-ContextualInformation	f1:84.44%	precision:44.97%	recall:58.69%	support:169
              Other-Discussion	f1:70.00%	precision:11.67%	recall:20.00%	support:300
              Other-Irrelevant	f1:74.82%	precision:39.16%	recall:51.41%	support:812
               Other-Sentiment	f1:83.22%	precision:43.06%	recall:56.76%	support:829
                         macro	f1:47.97%	precision:21.07%	recall:27.28%

# kick_off/only_text_encoder checkpoint: bert-base-uncased
Averaged stats: info_type_acc: 0.9269
===================================info_type====================================
         Request-GoodsServices	f1:0.00%  	precision:0.00%	  recall:0.00%	support:13
     Request-InformationWanted	f1:0.00%  	precision:0.00%	  recall:0.00%	support:27
       Request-SearchAndRescue	f1:0.00%	  precision:0.00%	  recall:0.00%	support:24
        CallToAction-Donations	f1:0.00%	  precision:0.00%	  recall:0.00%	support:96
       CallToAction-MovePeople	f1:0.00%	  precision:0.00%	  recall:0.00%	support:19
        CallToAction-Volunteer	f1:0.00%	  precision:0.00%	  recall:0.00%	support:12
                Report-CleanUp	f1:0.00%	  precision:0.00%	  recall:0.00%	support:11
        Report-EmergingThreats	f1:42.86%	  precision:7.41%	  recall:12.63%	support:162
                Report-Factoid	f1:78.44%	  precision:42.54%	recall:55.16%	support:590
  Report-FirstPartyObservation	f1:56.15%	  precision:16.86%	recall:25.93%	support:433
               Report-Hashtags	f1:81.62%	  precision:70.23%	recall:75.50%	support:917
               Report-Location	f1:76.86%	  precision:59.90%	recall:67.33%	support:621
        Report-MultimediaShare	f1:77.65%	  precision:32.94%	recall:46.26%	support:844
                   Report-News	f1:71.10%	  precision:49.10%	recall:58.09%	support:892
            Report-NewSubEvent	f1:0.00%	  precision:0.00%	  recall:0.00%	support:76
               Report-Official	f1:0.00%	  precision:0.00%	  recall:0.00%	support:140
          Report-OriginalEvent	f1:67.50%	  precision:9.75%	  recall:17.03%	support:277
       Report-ServiceAvailable	f1:0.00%	  precision:0.00%	  recall:0.00%	support:135
  Report-ThirdPartyObservation	f1:72.86%	  precision:50.00%	recall:59.30%	support:580
                Report-Weather	f1:67.59%	  precision:53.28%	recall:59.59%	support:274
                  Other-Advice	f1:0.00%	  precision:0.00%	  recall:0.00%	support:180
   Other-ContextualInformation	f1:82.57%	  precision:53.25%	recall:64.75%	support:169
              Other-Discussion	f1:68.42%	  precision:13.00%	recall:21.85%	support:300
              Other-Irrelevant	f1:74.53%	  precision:38.92%	recall:51.13%	support:812
               Other-Sentiment	f1:82.17%	  precision:41.13%	recall:54.82%	support:829
                         macro	f1:40.01%	  precision:21.53%	recall:26.78%
```

### Multi Task kick_off



* 输入：
  * `post_text`
  * `<posit_id>_0` 如果存在图片，否则0图
* 训练任务：
    * info_type_cls 多标签分类
    * priority_regression 优先级回归


debug
```shell
python finetuned_trecis.py \
  --config ./trecis/tiny.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --use_info_type_cls \
  --use_priority_regression

python finetuned_trecis.py \
  --config ./trecis/tiny.yaml \
  --output_dir output/trecis \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --device 'cuda:0' \
  --only_text_encoder \
  --use_info_type_cls \
  --use_priority_regression
```

experiment
```shell
# mtl_albef  ckpt(albef)  task(['itc', 'pr'])
python finetuned_trecis.py \
  --config ./trecis/mtl_albef.yaml \
  --output_dir output/trecis/mtl_albef_ONLYTEXT_False_CKPT_albef_TASK_itc_pr \
  --checkpoint ./provided_ckpt/ALBEF.pth \
  --save_eval_label \
  --use_info_type_cls \
  --use_priority_regression \
  --device 'cuda:0' > finetuning_log/mtl_albef_ONLYTEXT_False_CKPT_albef_TASK_itc_pr

# mtl_albef/only_text_encoder  ckpt(bert-base-uncased)  task(['itc', 'pr'])
python finetuned_trecis.py \
  --config ./trecis/mtl_albef.yaml \
  --output_dir output/trecis/mtl_albef_ONLYTEXT_True_CKPT_bert_TASK_itc_pr \
  --checkpoint '' \
  --save_eval_label \
  --only_text_encoder \
  --use_info_type_cls \
  --use_priority_regression \
  --device 'cuda:0' > finetuning_log/mtl_albef_ONLYTEXT_True_CKPT_bert_TASK_itc_pr
```

```record
# mtl_albef  ckpt(albef)  task(['itc', 'pr'])
# EPOCH 8
Averaged stats: info_type_cls_acc: 0.9275  priority_regression_acc: 0.5519
                              precision    recall  f1-score   support

       Request-GoodsServices     0.0000    0.0000    0.0000        13
   Request-InformationWanted     0.5000    0.0370    0.0690        27
     Request-SearchAndRescue     0.7857    0.4583    0.5789        24
      CallToAction-Donations     0.6866    0.4792    0.5644        96
     CallToAction-MovePeople     0.4545    0.2632    0.3333        19
      CallToAction-Volunteer     0.0000    0.0000    0.0000        12
              Report-CleanUp     1.0000    0.0909    0.1667        11
      Report-EmergingThreats     0.6310    0.3272    0.4309       162
              Report-Factoid     0.6812    0.5831    0.6283       590
Report-FirstPartyObservation     0.4643    0.3903    0.4241       433
             Report-Hashtags     0.7867    0.6957    0.7384       917
             Report-Location     0.7003    0.6924    0.6964       621
      Report-MultimediaShare     0.6657    0.5616    0.6093       844
                 Report-News     0.6772    0.6256    0.6503       892
          Report-NewSubEvent     0.3824    0.1711    0.2364        76
             Report-Official     0.4918    0.4286    0.4580       140
        Report-OriginalEvent     0.5389    0.3249    0.4054       277
     Report-ServiceAvailable     0.6000    0.4444    0.5106       135
Report-ThirdPartyObservation     0.6191    0.5466    0.5806       580
              Report-Weather     0.7269    0.6898    0.7079       274
                Other-Advice     0.4820    0.3722    0.4201       180
 Other-ContextualInformation     0.7951    0.5740    0.6667       169
            Other-Discussion     0.4764    0.3367    0.3945       300
            Other-Irrelevant     0.6790    0.4975    0.5743       812
             Other-Sentiment     0.7120    0.6562    0.6830       829

                   micro avg     0.6639    0.5540    0.6040      8433
                   macro avg     0.5815    0.4099    0.4611      8433
                weighted avg     0.6566    0.5540    0.5976      8433
                 samples avg     0.6195    0.5435    0.5576      8433

              precision    recall  f1-score   support

    Critical     0.1633    0.4000    0.2319        40
        High     0.3607    0.4219    0.3889       365
      Medium     0.9308    0.5785    0.7135      2465
         Low     0.2039    0.5284    0.2943       511

    accuracy                         0.5519      3381
   macro avg     0.4147    0.4822    0.4071      3381
weighted avg     0.7503    0.5519    0.6094      3381



# # mtl_albef/only_text_encoder  ckpt(bert-base-uncased)  task(['itc', 'pr'])
# EPOCH 8
Averaged stats: info_type_cls_acc: 0.9290  priority_regression_acc: 0.4933
                              precision    recall  f1-score   support

       Request-GoodsServices     0.6667    0.1538    0.2500        13
   Request-InformationWanted     0.5000    0.1481    0.2286        27
     Request-SearchAndRescue     0.7857    0.4583    0.5789        24
      CallToAction-Donations     0.6707    0.5729    0.6180        96
     CallToAction-MovePeople     0.4444    0.2105    0.2857        19
      CallToAction-Volunteer     0.0000    0.0000    0.0000        12
              Report-CleanUp     1.0000    0.0909    0.1667        11
      Report-EmergingThreats     0.6071    0.3148    0.4146       162
              Report-Factoid     0.6661    0.6254    0.6451       590
Report-FirstPartyObservation     0.5254    0.3349    0.4090       433
             Report-Hashtags     0.7621    0.7721    0.7671       917
             Report-Location     0.6801    0.7327    0.7054       621
      Report-MultimediaShare     0.6419    0.6457    0.6438       844
                 Report-News     0.6651    0.6300    0.6471       892
          Report-NewSubEvent     0.4783    0.1447    0.2222        76
             Report-Official     0.6129    0.4071    0.4893       140
        Report-OriginalEvent     0.5844    0.3249    0.4176       277
     Report-ServiceAvailable     0.6667    0.4889    0.5641       135
Report-ThirdPartyObservation     0.6485    0.5534    0.5972       580
              Report-Weather     0.7545    0.6058    0.6721       274
                Other-Advice     0.4528    0.4000    0.4248       180
 Other-ContextualInformation     0.7273    0.5680    0.6379       169
            Other-Discussion     0.5000    0.3000    0.3750       300
            Other-Irrelevant     0.6684    0.4766    0.5564       812
             Other-Sentiment     0.7291    0.6719    0.6993       829

                   micro avg     0.6682    0.5722    0.6165      8433
                   macro avg     0.6175    0.4253    0.4806      8433
                weighted avg     0.6591    0.5722    0.6063      8433
                 samples avg     0.6235    0.5576    0.5665      8433

              precision    recall  f1-score   support

    Critical     0.1579    0.3750    0.2222        40
        High     0.3632    0.4658    0.4082       365
      Medium     0.9367    0.4860    0.6400      2465
         Low     0.1852    0.5577    0.2780       511

    accuracy                         0.4933      3381
   macro avg     0.4107    0.4711    0.3871      3381
weighted avg     0.7520    0.4933    0.5553      3381
```