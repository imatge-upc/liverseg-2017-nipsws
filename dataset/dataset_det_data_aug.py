"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Class to define the Dataset object.
"""

from PIL import Image
import os
import numpy as np
import scipy.io


class Dataset:
    def __init__(self, train_list_pos, train_list_neg, val_list_pos, val_list_neg, test_list_pos, test_list_neg, database_root, store_memory=True):
        """Initialize the Dataset object
        Args:
        train_list: TXT file with the path to the images to use for training (Images must be between 0 and 255)
        test_list: TXT file with the path to the images to use for testing (Images must be between 0 and 255)
        database_root: Path to the root of the Database
        store_memory: True stores all the training images, False loads at runtime the images
        Returns:
        """
        
        # Load training images (path) and labels
        print('Started loading training files...')
        if train_list_pos is not None:
            with open(train_list_pos) as t:
                train_paths_positives = t.readlines()
        else:
            train_paths_positives = []
            
            
        if train_list_neg is not None:
            with open(train_list_neg) as t:
                train_paths_negatives = t.readlines()
        else:
            train_paths_negatives = []
            
            
        self.images_train_positives = []
        self.images_train_path_positives = []
        self.x_train_positives = []
        self.y_train_positives = []
        self.data_aug_id_positives_train = []
        
        self.labels_train_positives = []
        self.labels_train_path_positives = []
        self.x_train_negatives = []
        self.y_train_negatives = []
        self.data_aug_id_negatives_train = []
        
        self.images_train_negatives = []
        self.images_train_path_negatives = []
        self.labels_train_negatives = []
        self.labels_train_path_negatives = []

        
        for idx, line in enumerate(train_paths_positives):     
            self.images_train_path_positives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_train_positives.append(str(line.split()[1]))
            self.y_train_positives.append(str(line.split()[2]))
            self.data_aug_id_positives_train.append(str(line.split()[3]))
            self.labels_train_path_positives.append('1')
                
        self.images_train_path_positives = np.array(self.images_train_path_positives)
        self.labels_train_path_positives = np.array(self.labels_train_path_positives)
        self.x_train_positives = np.array(self.x_train_positives)
        self.y_train_positives = np.array(self.y_train_positives)
        
        for idx, line in enumerate(train_paths_negatives):     
            self.images_train_path_negatives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_train_negatives.append(str(line.split()[1]))
            self.y_train_negatives.append(str(line.split()[2]))
            self.data_aug_id_negatives_train.append(str(line.split()[3]))
            self.labels_train_path_negatives.append('0')
      
        self.images_train_path_negatives = np.array(self.images_train_path_negatives)
        self.labels_train_path_negatives = np.array(self.labels_train_path_negatives)
        self.x_train_negatives = np.array(self.x_train_negatives)
        self.y_train_negatives = np.array(self.y_train_negatives)
        self.data_aug_id_positives_train = np.array(self.data_aug_id_positives_train)
        self.data_aug_id_negatives_train = np.array(self.data_aug_id_negatives_train)

        # Load validation images (path) and labels
        print('Started loading validation files...')
        if val_list_pos is not None:
            with open(val_list_pos) as t:
                val_paths_positives = t.readlines()
        else:
            val_paths_positives = []
            
            
        if val_list_neg is not None:
            with open(val_list_neg) as t:
                val_paths_negatives = t.readlines()
        else:
            val_paths_negatives = []
            
            
        self.images_val_positives = []
        self.images_val_path_positives = []
        self.labels_val_positives = []
        self.labels_val_path_positives = []
        
        self.images_val_negatives = []
        self.images_val_path_negatives = []
        self.labels_val_negatives = []
        self.labels_val_path_negatives = []
        
        self.x_val_positives = []
        self.y_val_positives = []
        self.x_val_negatives = []
        self.y_val_negatives = []
        
        self.data_aug_id_positives_val = []
        self.data_aug_id_negatives_val = []
        
        for idx, line in enumerate(val_paths_positives):     
            self.images_val_path_positives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_val_positives.append(str(line.split()[1]))
            self.y_val_positives.append(str(line.split()[2]))
            self.data_aug_id_positives_val.append(str(line.split()[3]))
            self.labels_val_path_positives.append('1')
                
        self.images_val_path_positives = np.array(self.images_val_path_positives)
        self.labels_val_path_positives = np.array(self.labels_val_path_positives)
        self.x_val_positives = np.array(self.x_val_positives)
        self.y_val_positives = np.array(self.y_val_positives)
        self.data_aug_id_positives_val = np.array(self.data_aug_id_positives_val)
        
        for idx, line in enumerate(val_paths_negatives):     
            self.images_val_path_negatives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_val_negatives.append(str(line.split()[1]))
            self.y_val_negatives.append(str(line.split()[2]))
            self.data_aug_id_negatives_val.append(str(line.split()[3]))
            self.labels_val_path_negatives.append('0')
      
        self.images_val_path_negatives = np.array(self.images_val_path_negatives)
        self.labels_val_path_negatives = np.array(self.labels_val_path_negatives)
        self.x_val_negatives = np.array(self.x_val_negatives)
        self.y_val_negatives = np.array(self.y_val_negatives)
        self.data_aug_id_negatives_val = np.array(self.data_aug_id_negatives_val)
        
        # Load testing images (path) and labels
        print('Started loading testing files...')
        if test_list_pos is not None:
            with open(test_list_pos) as t:
                test_paths_positives = t.readlines()
        else:
            test_paths_positives = []
            
            
        if test_list_neg is not None:
            with open(test_list_neg) as t:
                test_paths_negatives = t.readlines()
        else:
            test_paths_negatives = []
            
            
        self.images_test_positives = []
        self.images_test_path_positives = []
        
        self.x_test_positives = []
        self.y_test_positives = []
        
        self.images_test_negatives = []
        self.images_test_path_negatives = []
        
        self.x_test_negatives = []
        self.y_test_negatives = []

        for idx, line in enumerate(test_paths_positives):     
            self.images_test_path_positives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_test_positives.append(str(line.split()[1]))
            self.y_test_positives.append(str(line.split()[2]))
                
        self.images_test_path_positives = np.array(self.images_test_path_positives)
        self.x_test_positives = np.array(self.x_test_positives)
        self.y_test_positives = np.array(self.y_test_positives)
        
        for idx, line in enumerate(test_paths_negatives):     
            self.images_test_path_negatives.append(os.path.join(database_root,str(line.split()[0])))
            self.x_test_negatives.append(str(line.split()[1]))
            self.y_test_negatives.append(str(line.split()[2]))
        
        self.images_test_path_negatives = np.array(self.images_test_path_negatives)
        self.x_test_negatives = np.array(self.x_test_negatives)
        self.y_test_negatives = np.array(self.y_test_negatives)
 
                
        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr_pos = 0
        self.train_ptr_neg = 0
        self.test_ptr_pos = 0
        self.test_ptr_neg = 0
        self.val_ptr_pos = 0
        self.val_ptr_neg = 0
        self.train_size_pos = len(self.images_train_path_positives)
        self.train_size_neg = len(self.images_train_path_negatives)
        self.val_size_pos = len(self.images_val_path_positives)
        self.val_size_neg = len(self.images_val_path_negatives)
        self.test_size_pos = len(self.images_test_path_positives)
        self.test_size_neg = len(self.images_test_path_negatives)        
        
        self.train_idx_pos = np.arange(self.train_size_pos)
        self.train_idx_neg = np.arange(self.train_size_neg)
        self.val_idx_pos = np.arange(self.val_size_pos)
        self.val_idx_neg = np.arange(self.val_size_neg)        
        
        np.random.shuffle(self.train_idx_pos)
#        np.random.shuffle(self.val_idx_pos)
        np.random.shuffle(self.train_idx_neg)
#        np.random.shuffle(self.val_idx_neg)
        
        self.store_memory = store_memory
        
    def next_batch(self, batch_size, phase, balance):
        
        positive_samples = int(balance*batch_size)
        negative_samples = int((1.0-balance)*batch_size)
        
        if phase == 'train':
            if self.train_ptr_pos + positive_samples < self.train_size_pos:
                idx = np.array(self.train_idx_pos[self.train_ptr_pos:self.train_ptr_pos + positive_samples])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                    x = [self.x_train_positives[l] for l in idx]
                    y = [self.y_train_positives[l] for l in idx]
                    ids = [self.data_aug_id_neg_train[l] for l in idx]
                else:
                    images = [self.images_train_path_positives[l] for l in idx]
                    labels = [self.labels_train_path_positives[l] for l in idx]
                    x = [self.x_train_positives[l] for l in idx]
                    y = [self.y_train_positives[l] for l in idx]
                    ids = [self.data_aug_id_negatives_train[l] for l in idx]
                self.train_ptr_pos += positive_samples
                images_pos = images
                labels_pos = labels
                x_pos = x
                y_pos = y
                ids_pos = ids
            else:
                old_idx = np.array(self.train_idx_pos[self.train_ptr_pos:])
                np.random.shuffle(self.train_idx_pos)
                new_ptr_pos = (self.train_ptr_pos + positive_samples) % self.train_size_pos
                idx = np.array(self.train_idx_pos[:new_ptr_pos])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                    x_1 = [self.x_train_positives[l] for l in old_idx]
                    y_1 = [self.y_train_positives[l] for l in old_idx]
                    x_2 = [self.x_train_positives[l] for l in idx]
                    y_2 = [self.y_train_positives[l] for l in idx]
                    ids_1 = [self.data_aug_id_positives_train[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_positives_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path_positives[l] for l in old_idx]
                    labels_1 = [self.labels_train_path_positives[l] for l in old_idx]
                    images_2 = [self.images_train_path_positives[l] for l in idx]
                    labels_2 = [self.labels_train_path_positives[l] for l in idx]
                    x_1 = [self.x_train_positives[l] for l in old_idx]
                    y_1 = [self.y_train_positives[l] for l in old_idx]
                    x_2 = [self.x_train_positives[l] for l in idx]
                    y_2 = [self.y_train_positives[l] for l in idx]
                    ids_1 = [self.data_aug_id_positives_train[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_positives_train[l] for l in idx]
                images_pos = images_1+images_2
                labels_pos = labels_1+labels_2
                x_pos = x_1 + x_2
                y_pos = y_1 + y_2
                ids_pos = ids_1 + ids_2
                self.train_ptr_pos = new_ptr_pos
            if self.train_ptr_neg + negative_samples < self.train_size_neg:
                idx = np.array(self.train_idx_neg[self.train_ptr_neg:self.train_ptr_neg + negative_samples])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                    x = [self.x_train_negatives[l] for l in idx]
                    y = [self.y_train_negatives[l] for l in idx]
                    ids = [self.data_aug_id_negatives_train[l] for l in idx]
                else:
                    images = [self.images_train_path_negatives[l] for l in idx]
                    labels = [self.labels_train_path_negatives[l] for l in idx]
                    x = [self.x_train_negatives[l] for l in idx]
                    y = [self.y_train_negatives[l] for l in idx]
                    ids = [self.data_aug_id_negatives_train[l] for l in idx]
                self.train_ptr_neg += negative_samples
                images_neg = images
                labels_neg = labels
                x_neg = x
                y_neg = y
                ids_neg = ids
            else:
                old_idx = np.array(self.train_idx_neg[self.train_ptr_neg:])
                np.random.shuffle(self.train_idx_neg)
                new_ptr_neg = (self.train_ptr_neg + negative_samples) % self.train_size_neg
                idx = np.array(self.train_idx_neg[:new_ptr_neg])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                    x_1 = [self.x_train_negatives[l] for l in old_idx]
                    y_1 = [self.y_train_negatives[l] for l in old_idx]
                    x_2 = [self.x_train_negatives[l] for l in idx]
                    y_2 = [self.y_train_negatives[l] for l in idx]
                    ids_1 = [self.data_aug_id_negatives_train[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_negatives_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path_negatives[l] for l in old_idx]
                    labels_1 = [self.labels_train_path_negatives[l] for l in old_idx]
                    images_2 = [self.images_train_path_negatives[l] for l in idx]
                    labels_2 = [self.labels_train_path_negatives[l] for l in idx]
                    x_1 = [self.x_train_negatives[l] for l in old_idx]
                    y_1 = [self.y_train_negatives[l] for l in old_idx]
                    x_2 = [self.x_train_negatives[l] for l in idx]
                    y_2 = [self.y_train_negatives[l] for l in idx]
                    ids_1 = [self.data_aug_id_negatives_train[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_negatives_train[l] for l in idx]
                images_neg = images_1+images_2
                labels_neg = labels_1+labels_2
                x_neg = x_1 + x_2
                y_neg = y_1 + y_2
                ids_neg = ids_1 + ids_2
                self.train_ptr_neg = new_ptr_neg                
            
            images = images_pos + images_neg
            labels = labels_pos + labels_neg
            x_bb = x_pos + x_neg
            y_bb = y_pos + y_neg
            ids_bb = ids_pos + ids_neg
            
        elif phase == 'val':
            if self.val_ptr_pos + positive_samples < self.val_size_pos:
                idx = np.array(self.val_idx_pos[self.val_ptr_pos:self.val_ptr_pos + positive_samples])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                    x = [self.x_val_positives[l] for l in idx]
                    y = [self.y_val_positives[l] for l in idx]
                    ids = [self.data_aug_id_positives_val[l] for l in idx]
                else:
                    images = [self.images_val_path_positives[l] for l in idx]
                    labels = [self.labels_val_path_positives[l] for l in idx]
                    x = [self.x_val_positives[l] for l in idx]
                    y = [self.y_val_positives[l] for l in idx]
                    ids = [self.data_aug_id_positives_val[l] for l in idx]
                self.val_ptr_pos += positive_samples
                images_pos = images
                labels_pos = labels
                x_pos = x
                y_pos = y
                ids_pos = ids
            else:
                old_idx = np.array(self.val_idx_pos[self.val_ptr_pos:])
#                np.random.shuffle(self.val_idx_pos)
                new_ptr_pos = (self.val_ptr_pos + positive_samples) % self.val_size_pos
                idx = np.array(self.val_idx_pos[:new_ptr_pos])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                    x_1 = [self.x_val_positives[l] for l in old_idx]
                    y_1 = [self.y_val_positives[l] for l in old_idx]
                    x_2 = [self.x_val_positives[l] for l in idx]
                    y_2 = [self.y_val_positives[l] for l in idx]
                    ids_1 = [self.data_aug_id_positives_val[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_positives_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path_positives[l] for l in old_idx]
                    labels_1 = [self.labels_val_path_positives[l] for l in old_idx]
                    images_2 = [self.images_val_path_positives[l] for l in idx]
                    labels_2 = [self.labels_val_path_positives[l] for l in idx]
                    x_1 = [self.x_val_positives[l] for l in old_idx]
                    y_1 = [self.y_val_positives[l] for l in old_idx]
                    x_2 = [self.x_val_positives[l] for l in idx]
                    y_2 = [self.y_val_positives[l] for l in idx]
                    ids_1 = [self.data_aug_id_positives_val[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_positives_val[l] for l in idx]
                images_pos = images_1+images_2
                labels_pos = labels_1+labels_2
                x_pos = x_1 + x_2
                y_pos = y_1 + y_2
                ids_pos = ids_1 + ids_2
                self.val_ptr_pos = new_ptr_pos
            if self.val_ptr_neg + negative_samples < self.val_size_neg:
                idx = np.array(self.val_idx_neg[self.val_ptr_neg:self.val_ptr_neg + negative_samples])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                    x = [self.x_val_negatives[l] for l in idx]
                    y = [self.y_val_negatives[l] for l in idx]
                    ids = [self.data_aug_id_negatives_val[l] for l in idx]
                else:
                    images = [self.images_val_path_negatives[l] for l in idx]
                    labels = [self.labels_val_path_negatives[l] for l in idx]
                    x = [self.x_val_negatives[l] for l in idx]
                    y = [self.y_val_negatives[l] for l in idx]
                    ids = [self.data_aug_id_negatives_val[l] for l in idx]
                self.val_ptr_neg += negative_samples
                images_neg = images
                labels_neg = labels
                x_neg = x
                y_neg = y
                ids_neg = ids
            else:
                old_idx = np.array(self.val_idx_neg[self.val_ptr_neg:])
#                np.random.shuffle(self.val_idx_neg)
                new_ptr_neg = (self.val_ptr_neg + negative_samples) % self.val_size_neg
                idx = np.array(self.val_idx_neg[:new_ptr_neg])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                    x_1 = [self.x_val_negatives[l] for l in old_idx]
                    y_1 = [self.y_val_negatives[l] for l in old_idx]
                    x_2 = [self.x_val_negatives[l] for l in idx]
                    y_2 = [self.y_val_negatives[l] for l in idx]
                    ids_1 = [self.data_aug_id_negatives_val[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_negatives_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path_negatives[l] for l in old_idx]
                    labels_1 = [self.labels_val_path_negatives[l] for l in old_idx]
                    images_2 = [self.images_val_path_negatives[l] for l in idx]
                    labels_2 = [self.labels_val_path_negatives[l] for l in idx]
                    x_1 = [self.x_val_negatives[l] for l in old_idx]
                    y_1 = [self.y_val_negatives[l] for l in old_idx]
                    x_2 = [self.x_val_negatives[l] for l in idx]
                    y_2 = [self.y_val_negatives[l] for l in idx]
                    ids_1 = [self.data_aug_id_negatives_val[l] for l in old_idx]
                    ids_2 = [self.data_aug_id_negatives_val[l] for l in idx]
                images_neg = images_1+images_2
                labels_neg = labels_1+labels_2
                x_neg = x_1 + x_2
                y_neg = y_1 + y_2
                ids_neg = ids_1 + ids_2
                self.val_ptr_neg = new_ptr_neg                
            
            images = images_pos + images_neg
            labels = labels_pos + labels_neg
            x_bb = x_pos + x_neg
            y_bb = y_pos + y_neg
            ids_bb = ids_pos + ids_neg
            
        elif phase == 'val_pos':
            if self.val_ptr_pos + 2*positive_samples < self.val_size_pos:
                idx = np.array(self.val_idx_pos[self.val_ptr_pos:self.val_ptr_pos + 2*positive_samples])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                else:
                    images = [self.images_val_path_positives[l] for l in idx]
                    labels = [self.labels_val_path_positives[l] for l in idx]
                self.val_ptr_pos += 2*positive_samples
                images_pos = images
                labels_pos = labels
            else:
                old_idx = np.array(self.val_idx_pos[self.val_ptr_pos:])
                np.random.shuffle(self.val_idx_pos)
                new_ptr_pos = (self.val_ptr_pos + 2*positive_samples) % self.val_size_pos
                idx = np.array(self.val_idx_pos[:new_ptr_pos])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path_positives[l] for l in old_idx]
                    labels_1 = [self.labels_val_path_positives[l] for l in old_idx]
                    images_2 = [self.images_val_path_positives[l] for l in idx]
                    labels_2 = [self.labels_val_path_positives[l] for l in idx]
                images_pos = images_1+images_2
                labels_pos = labels_1+labels_2
                self.val_ptr_pos = new_ptr_pos
            return images_pos, labels_pos
        elif phase == 'val_neg':
            if self.val_ptr_neg + 2*negative_samples < self.val_size_neg:
                idx = np.array(self.val_idx_neg[self.val_ptr_neg:self.val_ptr_neg + 2*negative_samples])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                else:
                    images = [self.images_val_path_negatives[l] for l in idx]
                    labels = [self.labels_val_path_negatives[l] for l in idx]
                self.val_ptr_neg += 2*negative_samples
                images_neg = images
                labels_neg = labels
            else:
                old_idx = np.array(self.val_idx_neg[self.val_ptr_neg:])
                np.random.shuffle(self.val_idx_neg)
                new_ptr_neg = (self.val_ptr_neg + 2*negative_samples) % self.val_size_neg
                idx = np.array(self.val_idx_neg[:new_ptr_neg])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path_negatives[l] for l in old_idx]
                    labels_1 = [self.labels_val_path_negatives[l] for l in old_idx]
                    images_2 = [self.images_val_path_negatives[l] for l in idx]
                    labels_2 = [self.labels_val_path_negatives[l] for l in idx]
                images_neg = images_1+images_2
                labels_neg = labels_1+labels_2
                self.val_ptr_neg = new_ptr_neg                
            
            return images_neg, labels_neg
            
        elif phase == 'test':
            if self.test_ptr_pos + positive_samples < self.test_size_pos:
                idx = np.array(self.test_idx_pos[self.test_ptr_pos:self.test_ptr_pos + positive_samples])
                if self.store_memory:
                    images = [self.images_test[l] for l in idx]
                else:
                    images = [self.images_test_path[l] for l in idx]
                self.test_ptr_pos += positive_samples
            else:
                old_idx = np.array(self.test_idx_pos[self.test_ptr_pos:])
                np.random.shuffle(self.test_idx_pos)
                new_ptr = (self.test_ptr_pos + positive_samples) % self.test_size_pos
                idx = np.array(self.test_idx_pos[:new_ptr_pos])
                if self.store_memory:
                    images_1 = [self.images_test[l] for l in old_idx]
                    images_2 = [self.images_test[l] for l in idx]
                else:
                    images_1 = [self.images_test_path[l] for l in old_idx]
                    images_2 = [self.images_test_path[l] for l in idx]
                images_pos = images_1+images_2
                self.test_ptr_pos = new_ptr_pos
            if self.test_ptr_neg + negative_samples < self.test_size_neg:
                idx = np.array(self.test_idx_neg[self.test_ptr_neg:self.test_ptr_neg + negative_samples])
                if self.store_memory:
                    images = [self.images_test[l] for l in idx]
                else:
                    images = [self.images_test_path[l] for l in idx]
                self.test_ptr_neg += negative_samples
            else:
                old_idx = np.array(self.test_idx_neg[self.test_ptr_neg:])
                np.random.shuffle(self.test_idx_neg)
                new_ptr = (self.test_ptr_neg + negative_samples) % self.test_size_neg
                idx = np.array(self.test_idx_neg[:new_ptr_neg])
                if self.store_memory:
                    images_1 = [self.images_test[l] for l in old_idx]
                    images_2 = [self.images_test[l] for l in idx]
                else:
                    images_1 = [self.images_test_path[l] for l in old_idx]
                    images_2 = [self.images_test_path[l] for l in idx]
                images_neg = images_1+images_2
                labels_neg = labels_1+labels_2
                self.test_ptr_neg = new_ptr_neg                
            
            images = images_pos + images_neg
            labels = labels_pos + labels_neg
            
        return images, labels, x_bb, y_bb, ids_bb
        

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size
        
    def get_val_pos_size(self):
        return self.val_size_pos
        
    def get_val_neg_size(self):
        return self.val_size_neg
        
    def get_val_size(self):
        return self.val_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width





