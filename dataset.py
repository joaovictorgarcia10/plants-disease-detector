import os
import shutil

def prepare_dataset():
    dataset_path = "dataset/PlantVillage"
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')

    # Cópia de Diretórios
    if not os.path.exists(train_dir):
        shutil.copytree(os.path.join(dataset_path, 'original_train'), train_dir)
  
    if not os.path.exists(val_dir):
        shutil.copytree(os.path.join(dataset_path, 'original_val'), val_dir)

    # Remoção de Diretórios Indesejados
    directories_to_remove = ['Blueberry___healthy', 'Cherry___healthy']

    for dir_name in directories_to_remove:
        train_dir_path = os.path.join(train_dir, dir_name)
        val_dir_path = os.path.join(val_dir, dir_name)
        if os.path.exists(train_dir_path):
            shutil.rmtree(train_dir_path)
        if os.path.exists(val_dir_path):
            shutil.rmtree(val_dir_path)

if __name__ == '__main__':
    prepare_dataset()