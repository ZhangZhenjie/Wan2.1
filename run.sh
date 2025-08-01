nohup torchrun --nproc_per_node=4 \
    generate.py \
    --offload_model True\
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image ../input.jpg \
    --dit_fsdp \
    --t5_cpu \
    --low_vram_mode \
    --ulysses_size 4 \
    --prompt "A pretty Asian girl sits elegantly in front of the camera. She smiles nicely, waves her hand and tries to kiss the camera, as if she sees her boyfriend." \
    --save_file ../output.mp4 \
    > wan.log 2>&1 &
