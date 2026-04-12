_base_ = [
    './segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py',
    '../_base_/datasets/isaid.py',
    '../_base_/schedules/schedule_160k.py',
    '../_base_/default_runtime.py'
]


model = dict(
    data_preprocessor=dict(
        size=(896,896)
    ),
    decode_head=dict(
        num_classes=15,
        loss_decode=dict(type='CrossEntropyLoss', avg_non_ignore=True)
    )  
)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=250, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000,save_best='mIoU')
)


#FP16
optimizer_config = dict(grad_clip=dict(max_norm=0.35, norm_type=2))
fp16 = dict(loss_scale='dynamic')
