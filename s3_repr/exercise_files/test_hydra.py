import hydra

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    epochs, batch_size = cfg.hyper.epochs, cfg.hyper.batch_size
    print(f'{epochs=}',f'{batch_size=}') 
          
if __name__ == "__main__":
    main()