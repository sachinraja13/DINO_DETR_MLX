_base_ = ['coco_transformer.py']

compile_computation_graph = True
compile_forward = True
# Will not help if compile_backward=True and compile_forward=False
compile_backward = False
device = 'gpu'  # 'gpu' or 'cpu'
precision = 'full'
quantize_model = False  # Support to quantize linear layers of the model
quantize_groups = 32
quantize_bits = 8

load_pytorch_weights = True
pytorch_weights_path = 'pytorch_weights/torch_model.pth'
resume_checkpoint = None  # Load from output_dir + resume_checkpoint directory
reset_optimizer = True

dataset_file = 'synthetic'
coco_year = '2017'
num_classes = 91

synthetic_image_size = (512, 512)
num_samples_synthetic_dataset = 1000
num_classes_synthetic_dataset = 91
min_targets_per_image = 50
max_targets_per_image = 200

pad_all_images_to_same_size = True  # Always set to False for evaluation
square_images = True
image_array_fixed_size = [512, 512, 3]
pad_labels_to_n_max_ground_truths = True  # Always set to False for evaluation
n_max_ground_truths = 500

use_custom_dataloader = False
reinstantiate_dataloader_every_epoch = False

optimizer_type = 'AdamW'
learning_schedule = 'cosine_decay'  # cosine_decay or step_decay
warm_up_learning_rate = True
warm_up_learning_rate_steps = 100
lr = 0.0001
param_dict_type = 'default'
batch_size = 1
weight_decay = 0.0001
epochs = 30
use_lr_drop_epochs = True  # If False, use lr_drop_steps
lr_drop_steps = 50000
lr_drop_epochs = 1
lr_drop_factor = 0.5
cosine_decay_num_epochs = 15
cosine_decay_end = 1e-9
save_checkpoint_interval = 1
clip_max_norm = 0.1
lr_drop_list = [33, 45]

print_freq = 10
print_loss_dict_freq = 5000

max_eval_iterations = None

modelname = 'dino'
frozen_weights = None
backbone = 'swin_large_patch4_window12'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperature = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False

# 'StableDINOCriterion' or 'DINOCriterion' or 'TwoStageCriterion' or 'BaseCriterion'
loss_criterion = 'StableDINOCriterion'
aux_loss = True
cost_class_type = 'focal_loss_cost'
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 6.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
focal_gamma = 2.0
two_stage_binary_cls = False
eos_coef = 0.1
ta_alpha = 0.0
ta_beta = 2.0
use_ce_loss_type = "stable-dino"
stg1_assigner = None
enc_kd_loss_weight = -1.0
enc_kd_loss_gamma = 2.0
target_post_process = "exp"

decoder_sa_type = 'sa'  # ['sa', 'ca_label', 'ca_content']
# or SimpleMinsumMatcher or HungarianMatcher or 'StableHungarianMatcher'
matcher_type = 'StableHungarianMatcher'
cec_beta = 0.5
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 91

match_unstable_error = True


use_detached_boxes_dec_out = False
