nohup torchrun --nproc_per_node=4 \
    generate.py \
    --offload_model True\
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image examples/i2v_input.JPG \
    --dit_fsdp \
    --t5_cpu \
    --low_vram_mode \
    --ulysses_size 4 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    > wan.log 2>&1 &
