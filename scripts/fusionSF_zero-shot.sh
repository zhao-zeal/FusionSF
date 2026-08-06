
  # Keep this conda environment isolated from incomplete packages installed in
  # ~/.local (notably regex and transformers), which otherwise shadow the
  # environment's own dependencies.
  export PYTHONNOUSERSITE=1
  export CUDA_VISIBLE_DEVICES=6

  python main.py \
    experiment=fusionsf_3modal_zeroshot \
    task_name=fusionsf_zeroshot_train10_19_test0_9 \
    seed=42 \
    resume=False \
    trainer.strategy=null \
    trainer.max_epochs=100 \
    callbacks.early_stopping.patience=35
