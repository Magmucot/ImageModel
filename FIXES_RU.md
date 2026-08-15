# Исправления после анализа Kaggle-лога

## Исправлено

- `VAE/train.py`: импорт модели изменён с несуществующего `VAE.models` на `VAE.vae`.
- `GAN/train.py`: импорт моделей изменён с несуществующего `GAN.models` на `GAN.gan`.
- VAE и GAN приведены к актуальному API `setup_distributed()` и `wrap_ddp()`.
- VAE и GAN теперь читают вложенные параметры YAML через `get_config_value()`.
- Параметры `latent_dim`, `ngf`, `ndf`, learning rate, Adam beta, epochs и output directory берутся из конфигов.
- Полное кэширование датасета в RAM больше не включается принудительно для каждого DDP-процесса.
- DDPM получил параметр `in_memory`; по умолчанию он выключен.
- DDPM в режиме `--dry_run` больше не запускает медленное 1000-шаговое семплирование.
- Добавлены регрессионные тесты для найденных ошибок.

## Проверка

```bash
python -m unittest discover -s tests -v
python -m py_compile VAE/train.py GAN/train.py DDPM/train.py
```

Перед обучением в Kaggle можно отдельно проверить импорты:

```bash
python -c "from VAE.vae import VAE; print('VAE import OK')"
python -c "from GAN.gan import Generator, Discriminator; print('GAN import OK')"
python -c "from DDPM.ddpm import DDPM, UNet; print('DDPM import OK')"
```

Полноценный GPU dry run следует выполнять в Kaggle, где установлены PyTorch и CUDA.
