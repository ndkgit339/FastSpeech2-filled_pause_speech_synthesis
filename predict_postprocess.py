import hydra
from omegaconf import DictConfig

from utils_predict import prepare_notalign, prepare_accent, \
                          concatenate_notaligned_data, copy_postprocessed_data

@hydra.main(config_path="config_predict", config_name="postprocess")
def main(config: DictConfig):

    prepare_notalign(config.data_dir, n_jobs=config.n_jobs)
    prepare_accent(config.data_dir)
    concatenate_notaligned_data(config.data_dir)
    copy_postprocessed_data(config.data_dir, config.target_preprocessed_dir,
                            config.target_raw_dir, config.speaker_name)
        
if __name__ == "__main__":
    main()
